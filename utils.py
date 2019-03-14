import torch

from dataloading import EOS_IDX, SOS_IDX


def prepare_batch(batch):
    # attach the opposite label to a batch
    x, lengths = batch.sent
    l = batch.label
    l_ = (l != 1).long()
    return (x, lengths), l, l_


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
    x, lengths = x # (B, L)
    lengths += 1
    B = x.size(0)
    if token == 'eos':
        eos = x.new_full((B,1), EOS_IDX)
        x = torch.cat([x, eos], dim=1)
    elif token == 'sos':
        sos = x.new_full((B,1), SOS_IDX)
        x = torch.cat([sos, x], dim=1)
    return (x, lengths)


def sequence_mask(lengths):
    # make a mask matrix corresponding to given length
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    row_vector = torch.arange(0, max(lengths), device=lengths.device) # (L,)
    matrix = lengths.unsqueeze(-1) # (B, 1)
    result = row_vector < matrix # 1 for real tokens
    return result # (B, L)


def get_actual_lengths(y):
    # get actual length of a generated batch considering eos
    non_zeros = (y == EOS_IDX).nonzero()
    num_nonzeros = non_zeros.size(0)
    is_dirty = [False for _ in range(num_nonzeros)]
    lengths = []
    for idx in non_zeros:
        i, j = idx[0].item(), idx[1].item()
        if not is_dirty[i]:
            is_dirty[i] = True
            lengths.append(j+1) # zero-index
    return torch.tensor(lengths, device=non_zeros.device)


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
    batch = [[vocab.itos[i] for i in ex] for ex in batch]
    return batch



