import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloading import PAD_IDX
from utils import reverse, kl_coef

logger = logging.getLogger(__name__)

class Stats(object):
    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.stats = {'recon_loss': [], 'kl_loss': []}

    def record_stats(self, recon_loss, kl_loss, stats=None):
        stats = self.stats if stats is None else stats
        stats['recon_loss'].append(recon_loss.item())
        stats['kl_loss'].append(kl_loss.item())

    # do not consider kl coef when reporting average of loss
    def report_stats(self, epoch, step=None, stats=None, is_train=True):
        stats = self.stats if stats is None else stats
        recon_loss = np.mean(stats['recon_loss'])
        kl_loss = np.mean(stats['kl_loss'])
        loss = recon_loss + kl_loss
        if is_train:
            msg = 'loss at epoch {}, step {}: {:.2f} ~ recon {:.2f} + kl {:.2f}'\
                .format(epoch, step, loss, recon_loss, kl_loss)
        else:
            msg = 'valid loss at epoch {}: {:.2f} ~ recon {:.2f} + kl {:.2f}'\
                .format(epoch, loss, recon_loss, kl_loss)
        logger.info(msg)


class Trainer(object):
    def __init__(self, model, data, lr=0.001):
        self.model = model
        self.data = data
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.stats = Stats()

    def _compute_loss(self, batch, total_step=None):
        logits, mu, log_var = self.model(batch.orig, batch.para)
        B, L, _ = logits.size()
        target, _ = batch.para
        recon_loss = self.criterion(logits.view(B*L, -1), target.view(-1))
        kl_loss = torch.sum((log_var - log_var.exp() - mu.pow(2) + 1)
                            * -0.5, dim=1).mean()
        coef = kl_coef(total_step) if total_step is not None else None # kl annlealing
        return recon_loss, kl_loss, coef

    # TODO: BOW loss
    def train(self, num_epoch):
        total_step = 0 # for KL annealing
        for epoch in range(num_epoch):
            self.stats.reset_stats()
            for step, batch in enumerate(self.data.train_iter, 1): # total 8280 step
                total_step += 1
                recon_loss, kl_loss, coef = self._compute_loss(batch, total_step)
                loss = recon_loss + coef * kl_loss
                self.stats.record_stats(recon_loss, kl_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if total_step % 100 == 0:
                    self.stats.report_stats(epoch, step=step)

            with torch.no_grad():
                valid_stats = {'recon_loss': [], 'kl_loss': []}
                for batch in self.data.valid_iter:
                    recon_loss, kl_loss, _= self._compute_loss(batch)
                    self.stats.record_stats(recon_loss, kl_loss, stats=valid_stats)
                self.stats.report_stats(epoch, stats=valid_stats, is_train=False)
                self.inference(data_iter=self.data.valid_iter)

    def inference(self, data_iter=None):
        if data_iter is not None: # for valid data
            data_iter.shuffle = True
            data_type = 'valid'
        else:
            data_iter = self.data.test_iter
            data_type = 'test'
        random_idx = random.randint(0, len(data_iter))
        for idx, batch in enumerate(data_iter): # to get a random batch
            if idx == random_idx: break
        paraphrased = self.model(batch.orig)
        original = reverse(batch.orig[0], self.data.vocab)
        paraphrased = reverse(paraphrased, self.data.vocab)
        print('sample paraphrases in {} data'.format(data_type))
        for orig, para in zip(original, paraphrased):
            print(orig, '\t => \t', para)

