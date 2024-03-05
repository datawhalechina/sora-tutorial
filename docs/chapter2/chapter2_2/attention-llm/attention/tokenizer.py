from collections import Counter
from typing import List

import jieba
import torch


class Tokenizer:

    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.special_tokens = ["[PAD]", "[UNK]"]
        self.unk_id = 1
        self.pad_id = 0
        self.vocab = None

    def load_vocab(self):
        assert self.vocab is not None
        self.vocab_size = len(self.vocab)
        self.word2id = dict(zip(self.vocab, range(self.vocab_size)))
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

    def tokenize(self, text: str) -> List[str]:
        res = []
        for token in jieba.cut(text):
            res.append(token)
        return res

    def encode(self, inputs: List[str]) -> List[List[int]]:
        res = []
        for s in inputs:
            tokens = self.tokenize(s)
            ids = [self.word2id.get(w, self.unk_id) for w in tokens]
            res.append(ids)
        return res

    def decode(self, ids: List[int]) -> str:
        res = []
        for i in ids:
            word = self.id2word.get(i)
            res.append(word)
        return "".join(res)
    
    def get_freq_of(self, word: str) -> int:
        return self.word_freq.get(word, 0)

    def build_vocab(self, text_list: list):
        self.vocab = self.special_tokens
        self.word_freq = {}
        words = []
        for text in text_list:
            for token in self.tokenize(text):
                words.append(token)
        count = Counter(words).most_common()
        for _i, (token, freq) in enumerate(count):
            if len(self.vocab) < self.vocab_size:
                self.vocab.append(token)
                self.word_freq[token] = freq
        self.load_vocab()

    def __call__(self, inp: List[str]):
        if isinstance(inp, str):
            inp = [inp]
        token_ids = self.encode(inp)
        ids = self.padding(token_ids)
        return torch.LongTensor(ids)

    def padding(self, token_ids: List[List[int]]) -> List[List[int]]:
        res = []
        max_len = max([len(v) for v in token_ids])
        max_len = min(max_len, self.max_seq_len)
        for i, sen_ids in enumerate(token_ids):
            length = len(sen_ids)
            if length < max_len:
                pad = [self.pad_id] * (max_len - length)
                sen_ids.extend(pad)
            else:
                sen_ids = sen_ids[:max_len]
            res.append(sen_ids)
        return res