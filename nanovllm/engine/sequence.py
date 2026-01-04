from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0 # 此序列有多少token被缓存
        self.block_table = [] # 此序列占用哪些block
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self): # 目前生成的token数，用于检测是否达到max_tokens
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self): # 获取prompt的token ids
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self): # 获取目前已生成的token_ids
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # 此序列占用了几个block
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # 计算num_tokens需要占用多少block
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self): # 占用的最后一个block里面存了多少token
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): # 取出某block对应的token_ids
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size] # python list的切片操作不会越界，如果最后一个block没有填满，会自动调整end

    def append_token(self, token_id: int): # 序列中添加一个token
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self): # 序列化
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state): # 反序列化
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
