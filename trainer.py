import logging
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nlgeval import NLGEval

from dataloading import PAD_IDX
from utils import reverse, kl_coef

logger = logging.getLogger(__name__)


class Stats():
    def __init__(self, to_record):
        self.to_record = to_record
        self.reset_stats()

    def reset_stats(self):
        self.stats = {name: [] for name in self.to_record}

    def record_stats(self, *args, stat=None):
        stats = self.stats if stat is None else stat
        for name, loss in zip(self.to_record, args):
            stats[name].append(loss.item())

    # do not consider kl coef when reporting average of loss
    def report_stats(self, epoch, step=None, stat=None):
        is_train = stat is None
        stats = self.stats if stat is None else stat
        losses = []
        for name in self.to_record:
            losses.append(np.mean(stats[name]))
        if is_train:
            msg = 'loss at epoch {} step {}: {:.2f} ~ recon {:.2f} + kl {:.2f} '\
                .format(epoch, step, losses[0]+losses[1], losses[0], losses[1])
        else:
            msg = 'valid loss at epoch {}: {:.2f} ~ recon {:.2f} + kl {:.2f}'\
                .format(epoch, losses[0]+losses[1], losses[0], losses[1])
        logger.info(msg)


class EarlyStopper():
    def __init__(self, patience, metric):
        self.patience = patience
        self.metric = metric # 'Bleu_1', ..., 'METEOR', 'ROUGE_L'
        self.count = 0
        self.best_score = defaultdict(lambda: 0)

    def stop(self, cur_score):
        if self.best_score[self.metric] > cur_score[self.metric]:
            if self.count <= self.patience:
                self.count += 1
                logger.info('Counting early stop patience... {}'.format(self.count))
                return False
            else:
                logger.info('Early stopping patience exceeded. Stopping training...')
                return True # halt training
        else:
            self.count = 0
            self.best_score = cur_score
            return False


class Trainer():
    def __init__(self, model, data, lr=0.001, to_record=None, clip=5, patience=3,
                 metric='Bleu_1'):
        self.model = model
        self.data = data
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.stats = Stats(to_record)
        self.early_stopper = EarlyStopper(patience, metric)
        self.evaluator = NLGEval(no_skipthoughts=True, no_glove=True)

    def _compute_loss(self, batch, total_step=0):
        logits, mu, log_var = self.model(batch.orig, batch.para)
        B, L, _ = logits.size()
        target, _ = batch.para
        recon_loss = self.criterion(logits.view(B*L, -1), target.view(-1))
        kl_loss = torch.sum((log_var - log_var.exp() - mu.pow(2) + 1)
                            * -0.5, dim=1).mean()
        coef = kl_coef(total_step) # kl annlealing
        return recon_loss, kl_loss, coef

    def train(self, num_epoch):
        total_step = 0 # for KL annealing
        for epoch in range(1, num_epoch+1, 1):
            self.stats.reset_stats()
            for step, batch in enumerate(self.data.train_iter, 1): # total 8280 step
                total_step += 1
                recon_loss, kl_loss, coef = self._compute_loss(batch, total_step)
                loss = recon_loss + coef * kl_loss
                self.stats.record_stats(recon_loss, kl_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()

            # report at the end of every epoch
            self.stats.report_stats(epoch, step=step)
            # DEBUG
            #train_metrics = self.evaluate(data_type='train')
            # evaluate at the end of every epoch
            with torch.no_grad():
                valid_stats = {name: [] for name in self.stats.to_record}
                for batch in self.data.valid_iter:
                    recon_loss, kl_loss, _= self._compute_loss(batch)
                    self.stats.record_stats(recon_loss, kl_loss, stat=valid_stats)
                self.stats.report_stats(epoch, stat=valid_stats)
                valid_metrics = self.evaluate(data_type='valid')
            # early stopping check
            if self.early_stopper.stop(valid_metrics):
                self.model.load_state_dict(best_model)
                # TODO: check if loading succeeds
                print('done')
            else:
                best_model = deepcopy(self.model.state_dict())
        # TODO: save model to a file
        logger.info('QUANTITATIVE TRAINING RESULTS: ', self.early_stopper.best_score)
        # results on test data at the end of training
        test_metrics = self.evaluate(data_type='test')

    def evaluate(self, data_type):
        data_iter = getattr(self.data, '{}_iter'.format(data_type))
        paraphrased, original, reference = [], [], []
        # TODO: valid batch size
        for idx, batch in enumerate(data_iter):
            para = self.model.inference(batch.orig)
            paraphrased += reverse(para, self.data.vocab)
            original += reverse(batch.orig[0], self.data.vocab)
            reference += reverse(batch.para[0], self.data.vocab)
        # for qualitative evaluation
        print('sample paraphrases in {} data'.format(data_type))
        for _ in range(data_iter.batch_size):
            random_idx = random.randint(0, len(data_iter))
            print(original[random_idx], '\t => \t', paraphrased[random_idx])
            print('\t\t\t reference: ', reference[random_idx])
        # for quantitative evaluation
        metrics_dict = self.evaluator.compute_metrics([reference], paraphrased)
        print('quantitative results from {} data'.format(data_type))
        print(metrics_dict)
        return metrics_dict




class Trainer_BOW(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_loss(self, batch, total_step=0): # overriding
        logits, mu, log_var, bow_logits = self.model(batch.orig, batch.para)
        B, L, _ = logits.size()
        target, _ = batch.para # (B, L)
        num_tokens = (target != PAD_IDX).sum().float()
        mask = torch.ones_like(bow_logits)
        mask[:, PAD_IDX] = 0

        recon_loss = self.criterion(logits.view(B*L, -1), target.view(-1))
        kl_loss = torch.sum((log_var - log_var.exp() - mu.pow(2) + 1) * -0.5, dim=1).mean()
        coef = kl_coef(total_step) # kl annlealing
        bow_loss = (bow_logits * mask).log_softmax(dim=-1).gather(1, target).sum() * -1 / num_tokens
        return recon_loss, kl_loss, coef, bow_loss

    def train(self, num_epoch):
        total_step = 0 # for KL annealing
        for epoch in range(1, num_epoch, 1):
            # TODO: running average?
            #self.stats.reset_stats()
            for step, batch in enumerate(self.data.train_iter, 1): # total 8280 step
                total_step += 1
                recon_loss, kl_loss, coef, bow_loss = self._compute_loss(batch, total_step)
                loss = recon_loss + coef * kl_loss + bow_loss
                self.stats.record_stats(recon_loss, kl_loss, bow_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 1000 == 0:
                    self.stats.report_stats(epoch, step=step)

            with torch.no_grad():
                valid_stats = {name: [] for name in self.stats.to_record}
                for batch in self.data.valid_iter:
                    recon_loss, kl_loss, _, bow_loss= self._compute_loss(batch)
                    self.stats.record_stats(recon_loss, kl_loss, bow_loss,
                                            stat=valid_stats)
                self.stats.report_stats(epoch, stat=valid_stats)
                self.inference(data_iter=self.data.valid_iter)


