import os
import torch
import logging
from torchtext.data import Field, TabularDataset, BucketIterator


MAXLEN = 15
logger = logging.getLogger(__name__)


# TODO: sos, eos
class Data(object):
    def __init__(self, data_dir, file):
        self.name = file
        data_dir =  os.path.join(data_dir, file)
        self.train_path = os.path.join(data_dir, 'train.txt')
        self.test_path = os.path.join(data_dir, 'test.txt')
        self.build()

    def build(self):
        self.src_field, self.trg_field = self.build_field(maxlen=MAXLEN)
        logger.info('building datasets... this takes a while')
        self.train, self.val, self.test =\
            self.build_dataset(self.src_field, self.trg_field)
        self.vocab = self.build_vocab(self.src_field, self.trg_field,
                                      self.train.src, self.train.trg,
                                      self.val.src, self.val.trg)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.val, self.test)
        logger.info('data size... {} / {} / {}'.format(len(self.train),
                                                       len(self.val),
                                                       len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build_field(self, maxlen=None):
        src_field= Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1])
        trg_field = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1])
        return src_field, trg_field

    def build_dataset(self, src_field, trg_field):
        train_val = TabularDataset(path=self.train_path, format='tsv',
                                fields=[('src', src_field),
                                        ('trg', trg_field)])
        train, val = train_val.split(split_ratio=0.8)
        # FIXME: test data is too large!
        test = TabularDataset(path=self.test_path, format='tsv',
                                fields=[('src', src_field),
                                        ('trg', trg_field)])
        return train, val, test

    def build_vocab(self, src_field, trg_field, *args):
        # not using pretrained word vectors
        src_field.build_vocab(args, max_size=30000)
        trg_field.vocab = src_field.vocab
        return src_field.vocab

    def build_iterator(self, train, val, test):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, val, test), batch_size=32,
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              sort_within_batch=True, repeat=False,
                              device=torch.device('cuda'))
        return train_iter, valid_iter, test_iter



