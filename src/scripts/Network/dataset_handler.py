import sys

sys.path.append('../.')
import CSVHelper
import JSONHelper


def get_dataset(split_file):
    rows = CSVHelper.read_as_dict(split_file)

    train_lst = []
    test_lst = []
    val_lst = []

    for data in rows:
        if data['split'] == 'train':
            train_lst.append({'synset_id': '0'+data['synsetId'], 'model_id': data['modelId']})
        elif data['split'] == 'val':
            val_lst.append({'synset_id': '0'+data['synsetId'], 'model_id': data['modelId']})
        elif data['split'] == 'test':
            test_lst.append({'synset_id': '0'+data['synsetId'], 'model_id': data['modelId']})

    return train_lst, val_lst, test_lst


if __name__ == '__main__':
    params = JSONHelper.read('../parameters.json')
    split_file = params['shapenet'] + 'dataset_split.csv'

    get_dataset(split_file)
