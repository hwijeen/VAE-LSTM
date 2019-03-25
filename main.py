import logging
from setproctitle import setproctitle

import torch

from dataloading import Data
from model import build_VAELSTM
from trainer import Trainer


DATA_DIR = '/home/nlpgpu5/hwijeen/VAE-LSTM/data/'
FILE = 'mscoco'
DEVICE = torch.device('cuda:0')


setproctitle("(hwijeen) VAE-LSTM")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    data = Data(DATA_DIR, FILE)
    vaeLSTM = build_VAELSTM(len(data.vocab), hidden_dim=600, latent_dim=1100,
                            word_drop=0.5, device=DEVICE)
    trainer = Trainer(vaeLSTM, data, lr=0.001)

    trainer.train(num_epoch=10)

