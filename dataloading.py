import os
import torch
import logging
from torchtext.data import Field, TabularDataset, BucketIterator


MAXLEN = 15
logger = logging.getLogger(__name__)

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Data(object):
    def __init__(self, data_dir, file):
        self.name = file
        data_dir =  os.path.join(data_dir, file)
        self.train_path = os.path.join(data_dir, 'train.txt')
        self.test_path = os.path.join(data_dir, 'test.txt')
        self.build()

    def build(self):
        self.ORIG, self.PARA = self.build_field(maxlen=MAXLEN)
        logger.info('building datasets... this takes a while')
        self.train, self.val, self.test =\
            self.build_dataset(self.ORIG, self.PARA)
        self.vocab = self.build_vocab(self.ORIG, self.PARA,
                                      self.train.orig, self.train.para,
                                      self.val.orig, self.val.para)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.val, self.test)
        logger.info('data size... {} / {} / {}'.format(len(self.train),
                                                       len(self.val),
                                                       len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build_field(self, maxlen=None):
        ORIG = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>')
        PARA = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>')
        return ORIG, PARA

    def build_dataset(self, ORIG, PARA):
        train_val = TabularDataset(path=self.train_path, format='tsv',
                                fields=[('orig', ORIG),
                                        ('para', PARA)])
        train, val = train_val.split(split_ratio=0.8)
        # FIXME: test data is too large!
        test = TabularDataset(path=self.test_path, format='tsv',
                                fields=[('orig', ORIG),
                                        ('para', PARA)])
        return train, val, test

    # TODO: add sos token
    def build_vocab(self, ORIG, PARA, *args):
        # not using pretrained word vectors
        ORIG.build_vocab(args, max_size=30000)
        ORIG.vocab.itos.insert(2, '<sos>')
        from collections import defaultdict
        stoi = defaultdict(lambda x:0)
        stoi.update({tok: i for i, tok in enumerate(ORIG.vocab.itos)})
        ORIG.vocab.stoi = stoi
        PARA.vocab = ORIG.vocab
        return ORIG.vocab

    def build_iterator(self, train, val, test):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, val, test), batch_size=32,
                              sort_key=lambda x: (len(x.orig), len(x.para)),
                              sort_within_batch=True, repeat=False,
                              device=torch.device('cuda'))
        return train_iter, valid_iter, test_iter



