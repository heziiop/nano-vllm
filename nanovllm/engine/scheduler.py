from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # waiting和running都为空时，表示没有正在执行的seq
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新的seq加到waiting队列
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 如果num_batched_tokens超限或者kvcache分不出来，那这次就取这么多seq去做prefill
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq) # 为seq分配/复用block_id
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 加上未命中前缀缓存的token数
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs: # 如果没有seq能进行prefill，改为执行decode
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) # 倾向于抢占一个后进的seq
                else:
                    # 抢占自己，因为running队列已为空，没有资源给此seq做decode
                    # 已从running队列中移出，干脆抢占自己，而不是加回到running队列
                    self.preempt(seq)
                    break
            else:
                # 如果是因为可以append才走到这里
                num_seqs += 1
                self.block_manager.may_append(seq) # 可能分配一个新block或者更新block
                scheduled_seqs.append(seq)
        assert scheduled_seqs # decode 阶段至少要调度到 1 个 seq，否则说明逻辑有问题
        self.running.extendleft(reversed(scheduled_seqs)) # 把刚刚选取的 seq 放回 running 队列的左边，保持它们的顺序不变
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 将此seq的状态从running改为waiting，并释放其占用的block,放到waiting队列左侧以便后续优先调度
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        # 后处理
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
