import logging

import torch
import torch.nn as nn
import torch.optim as optim

from dataloading import PAD_IDX
from utils import prepare_batch, reverse

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = optim.Adam(model.parameters())

    # TODO: logging and tqdm
    def train(self, epoch):
        for i in range(epoch):
            for step, batch in enumerate(self.data.train_iter, 1):
                (logits, _), mu, log_var = self.model(batch.orig, batch.para)
                B, L, _ = logits.size()
                target, _ = batch.para

                recon_loss = self.criterion(logits.view(B*L, -1),
                                            target.view(-1))
                kl_loss = -0.5 * (log_var - log_var.exp() - mu.pow(2) + 1).sum()
                loss = recon_loss + kl_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    msg = 'loss at epoch {}, step {}: {:.2f} = ' \
                          'recon {:.2f} +  kl{:.2f}'\
                        .format(i, step, loss, recon_loss, kl_loss)
                    logger.info(msg)
            # TODO: implement evaluation(inference)
                if step % 1000 == 0:
                    #self.evaluate()
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
