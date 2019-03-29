import math
import torch

from dataloading import EOS_IDX, SOS_IDX, UNK_IDX


def truncate(x, token=None):
    # delete a special token in a batch
    assert token in ['sos', 'eos', 'both'], 'can only truncate sos or eos'
    x, lengths = x # (B, L)
    lengths -= 1
    if token == 'sos': x = x[:, 1:]
    elif token == 'eos': x = x[:, :-1]
    else: x = x[:, 1:-1]
    return (x, lengths)


def append(x, token=None):
    # add a special token to a batch
    assert token in ['sos', 'eos'], 'can only append sos or eos'
    x, lengths = x # (B, L), (B,)
    lengths += 1
    B = x.size(0)
    if token == 'eos':
        eos = x.new_full((B,1), EOS_IDX)
        x = torch.cat([x, eos], dim=1)
    elif token == 'sos':
        sos = x.new_full((B,1), SOS_IDX)
        x = torch.cat([sos, x], dim=1)
    return (x, lengths)


def reverse(batch, vocab):
    # turn a batch of idx to tokens
    batch = batch.tolist()

    def trim(s, t):
        sentence = []
        for w in s:
            if w == t:
                break
            sentence.append(w)
        return sentence
    batch = [trim(ex, EOS_IDX) for ex in batch]
    batch = [' '.join([vocab.itos[i] for i in ex]) for ex in batch]
    return batch


def kl_coef(i):
    # coef for KL annealing
    # reaches 1 at i = 22000
    # https://github.com/kefirski/pytorch_RVAE/blob/master/utils/functional.py
    return (math.tanh((i - 3500)/1000) + 1) / 2


def word_drop(x, p):
    # p is prob to drop
    mask = torch.empty_like(x).bernoulli_(p).byte()
    x.masked_fill_(mask, UNK_IDX)
    return x

