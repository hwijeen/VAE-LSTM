import json
from collections import defaultdict

DATA_DIR = '/home/nlpgpu5/hwijeen/data/annotations'
DATA_TYPE = ['train2014', 'val2014']
OUTPUT_DIR = 'data/MSCOCO.txt' 

def load_annotations(data_dir, data_type):
    anns = defaultdict(list) 
    train_data = '{}/captions_{}.json'.format(DATA_DIR, DATA_TYPE[0])
    val_data = '{}/captions_{}.json'.format(DATA_DIR, DATA_TYPE[1])

    dataset_train = json.load(open(train_data, 'r')) 
    dataset_val = json.load(open(val_data, 'r'))
    print('train / val size: {} / {}'.format(
        len(dataset_train['images']), len(dataset_val['images'])))

    for dataset in [dataset_train, dataset_val]:
        for ann in dataset['annotations']:
            anns[ann['image_id']].append(ann['caption'])
    return anns

def extract_pairs(anns):
    print(len(anns))
    input()
    pairs = []
    for k, v in anns.items():
        s1, s2, s3, s4 = v[:4]
        pairs.extend([(s1, s2), (s3, s4)])
    return pairs

def save_to_file(paraphrases):
    print('saving {} paraphrase pairs to a file in {}'.format(
    len(paraphrases), OUTPUT_DIR))
    with open(OUTPUT_DIR, 'w') as f:
        for ori, para in paraphrases:
            f.write('{}\t{}\n'.format(ori, para))

if __name__ == "__main__":

    annotations = load_annotations(DATA_DIR, DATA_TYPE)
    paraphrases = extract_pairs(annotations)
    save_to_file(paraphrases)
