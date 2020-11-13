import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data

TRAINDATA_DIR = './train/'
TESTDATA_PATH = './test/testing-X.pkl'
ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


class CompDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self._data = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def extract_features(data, has_label=True):
    data['SimillarHTTP'] = 0.
    if has_label:
        return data.iloc[:, -80:-1]

    return data.iloc[:, -79:]


class UserRoundData(object):
    def __init__(self):
        self.data_dir = TRAINDATA_DIR
        self._user_datasets = []
        self.attack_types = ATTACK_TYPES
        self._load_data()

    def cal(self, x):
        if not x: return 0
        try:
            return float(x)
        except:
            return 13. if re.match(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}/?.{0,5}\.?.{0,10}\??', x) else 0.
    
    def normalize_1000(self, data):
        mi = data.min(0, keepdim=True)[0]
        ma = data.max(0, keepdim=True)[0]
        data -= mi
        data /= ma + 1e-9
        data *= 1000.

    def normalize_dy(self):
        for u in self._user_datasets:
            u = [torch.Tensor(u[0]), u[1]]
            self.normalize_1000(u[0])

    # dy: make 'similarhttp' useful
    def _get_data(self, fname):
        print('Reading: ' + fname)
        cont = open(fname).readlines()[1:]
        cont = [d[:-1].split(',')[-80:] for d in cont]
        x = [list(map(self.cal, i[:-1])) for i in cont]
        x = np.array(x).astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        y = np.array([self.attack_types[i[-1].split('_')[-1].replace('-', '').lower()] for i in cont])
        return (x, y)

    def _get_data_ori(self, fpath):
        if not fpath.endswith('csv'):
            return

        print('Load User Data: ', os.path.basename(fpath))
        data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
        x = extract_features(data)
        y = np.array([
            self.attack_types[t.split('_')[-1].replace('-', '').lower()]
            for t in data.iloc[:, -1]
        ])

        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        return (
            x,
            y,
        )

    def _load_data(self):
        _user_datasets = []
        self._user_datasets = []
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in [f for f in fnames if f.endswith('csv')]:
                data = self._get_data(os.path.join(root, fname))
                if data is not None:
                    _user_datasets.append(data)

        for x, y in _user_datasets:
            self._user_datasets.append((
                x,
                y,
            ))

        self.n_users = len(_user_datasets)
        self.pt_users = {}
        for u in range(self.n_users): self.pt_users[u] = 0
            
    # dy: cut data into pieces and shuffle when restart
    def round_data(self, user_idx, n_round, n_round_samples=-1):
        if n_round_samples == -1: return self._user_datasets[user_idx]
        
        n = len(self._user_datasets[user_idx][1])
        # at least 'n_round_samples' data for batchnorm
        if self.pt_users[user_idx] + n_round_samples >= n:
            self.pt_users[user_idx] = 0
            state = np.random.get_state()
            np.random.shuffle(self._user_datasets[user_idx][0])
            np.random.set_state(state)
            np.random.shuffle(self._user_datasets[user_idx][1])
        i = self.pt_users[user_idx]
        j = i + n_round_samples
        self.pt_users[user_idx] = j
        return self._user_datasets[user_idx][0][i:j], self._user_datasets[user_idx][1][i:j]

    def round_data_ori(self, user_idx, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            user_idx: int,  in [0, self.n_users)
            n_round: int, round number
        """
        if n_round_samples == -1:
            return self._user_datasets[user_idx]

        n_samples = len(self._user_datasets[user_idx][1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self._user_datasets[user_idx][0][choices], self._user_datasets[
            user_idx][1][choices]

    def uniform_random_loader(self, n_samples, batch_size=1000):
        X, Y = [], []
        n_samples_each_user = n_samples // len(self._user_datasets)
        if n_samples_each_user <= 0:
            n_samples_each_user = 1

        for idx in range(len(self._user_datasets)):
            x, y = self.round_data(user_idx=idx,
                                   n_round=0,
                                   n_round_samples=n_samples_each_user)
            X.append(x)
            Y.append(y)

        data = CompDataset(X=np.concatenate(X), Y=np.concatenate(Y))
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=min(batch_size, n_samples),
            shuffle=True,
        )

        return train_loader


def get_test_loader(batch_size=1000):
    with open(TESTDATA_PATH, 'rb') as fin:
        data = pickle.load(fin)

    test_loader = torch.utils.data.DataLoader(
        data['X'],
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader
