import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloading import PAD_IDX
from utils import prepare_batch, reverse, kl_coef

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

    # do not consider kl coef when reporting running average of loss
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

    def _compute_loss(self, batch, total_step):
        (logits, _), mu, log_var = self.model(batch.orig, batch.para)
        B, L, _ = logits.size()
        target, _ = batch.para
        recon_loss = self.criterion(logits.view(B*L, -1), target.view(-1))
        kl_loss = torch.sum((log_var - log_var.exp() - mu.pow(2) + 1)
                            * -0.5, dim=1).mean() 
        coef = kl_coef(total_step) # kl annlealing 
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
                    recon_loss, kl_loss = self._compute_loss(batch)
                    self.stats.record_stats(stats=valid_stats)
                self.stats.report_stats(epoch, stats=valid_stats, is_train=False)

    def inference(self):
        pass


    # TODO: early stopping
#    def evaluate(self):
#        self.data.valid_iter.shuffle = True
#        import random
#        a = random.randint(0, len(self.data.valid_iter)) # temp test
#        for i, batch in enumerate(self.data.valid_iter):
#            if i != a: continue
#            (x, lengths), l, l_ = prepare_batch(batch)
#            generated = self.model((x, lengths), l, l_, is_gen=True)
#            print('=' * 50)
#            print('original \t\t -> \t\t changed')
#            for idx in random.sample(range(lengths.size(0)), 5):
#                ori = reverse(x, self.data.vocab)[idx]
#                chg = reverse(generated[0], self.data.vocab)[idx]
#                print(' '.join(ori))
#                print('\t\t->', ' '.join(chg))
#            print('=' * 50)
#            return

    #def decode(self, gen_logit):
        #pass

    #def inference(self, pos_test_iter, neg_test_iter):
    #    pass
