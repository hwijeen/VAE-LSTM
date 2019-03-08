import logging
from setproctitle import setproctitle

from dataloading import Data


DATA_DIR = '/home/nlpgpu5/hwijeen/Paraphrase/data/'
FILE = 'mscoco'


setproctitle("(paraphrase) testing")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    logger.info('loaded data from... {}{}'.format(DATA_DIR, FILE))
    data = Data(DATA_DIR, FILE)

    #print(data.vocab.itos[:5])
