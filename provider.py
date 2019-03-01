# everything related to data provider
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing.label import LabelEncoder
from src.sharedCode.provider import *

def _parameters():
    return \
    {
        'data_path': None,
        'epochs': 300,
        'momentum': 0.7,
        'lr_start': 0.1,
        'lr_ep_step': 20,
        'lr_adaption': 0.5,
        'test_ratio': 0.1,
        'batch_size': 128,
        'cuda': False
    }


class PersistenceDiagramProviderCollate:
    def __init__(self, provider, wanted_views: [str] = None,
                 label_map: callable = lambda x: x,
                 output_type=torch.FloatTensor,
                 target_type=torch.LongTensor):
        provided_views = provider.view_names

        if wanted_views is None:
            self.wanted_views = provided_views

        else:
            for wv in wanted_views:
                if wv not in provided_views:
                    raise ValueError('{} is not provided by {} which provides {}'.format(wv, provider, provided_views))

            self.wanted_views = wanted_views

        if not callable(label_map):
            raise ValueError('label_map is expected to be callable.')

        self.label_map = label_map

        self.output_type = output_type
        self.target_type = target_type

    def __call__(self, sample_target_iter):
        batch_views_unprepared, batch_views_prepared, targets = defaultdict(list), {}, []

        for dgm_dict, label in sample_target_iter:
            for view_name in self.wanted_views:
                dgm = list(dgm_dict[view_name])
                dgm = self.output_type(dgm)

                batch_views_unprepared[view_name].append(dgm)

            targets.append(self.label_map(label))

        targets = self.target_type(targets)

        return batch_views_unprepared, targets


class SubsetRandomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def train_test_from_dataset(dataset,
                            test_size=0.2,
                            batch_size=16,
                            wanted_views=None):

    sample_labels = list(dataset.sample_labels)
    label_encoder = LabelEncoder().fit(sample_labels)
    sample_labels = label_encoder.transform(sample_labels)

    label_map = lambda l: int(label_encoder.transform([l])[0])
    collate_fn = PersistenceDiagramProviderCollate(dataset, label_map=label_map, wanted_views=wanted_views)

    sp = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_i, test_i = list(sp.split([0]*len(sample_labels), sample_labels))[0]

    data_train = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False,
                            sampler=SubsetRandomSampler(train_i.tolist()))

    data_test = DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           shuffle=False,
                           sampler=SubsetRandomSampler(test_i.tolist()))

    return data_train, data_test


def _data_setup(params):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(32)])
    assert (str(len(subscripted_views)) in params['data_path'])

    print('Loading provider...')
    dataset = Provider()
    dataset.read_from_h5(params['data_path'])

    assert all(view_name in dataset.view_names for view_name in subscripted_views)

    print('Create data loader...')
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=params['test_ratio'],
                                                    batch_size=params['batch_size'])

    return data_train, data_test, subscripted_views