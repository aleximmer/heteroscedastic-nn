import os
import numpy as np
import sklearn.model_selection as modsel
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as data

CRISPR_DATASETS = [
    'flow-cytometry-HEK293', 'survival-screen-A375', 'survival-screen-HEK293'
]


class CrisprDatasets(data.Dataset):
    def __init__(self, data_set, split='train', split_train_size=0.9, split_valid_size=0.1,
                 seed=6, shuffle=True, root='data/', scaling=True, double=False, rep=0):

        assert isinstance(seed, int), 'Please provide an integer random seed'
        error_msg = 'invalid UCI regression dataset'
        assert data_set in CRISPR_DATASETS, error_msg
        assert 0.0 <= split_train_size <= 1.0, 'split_train_size does not lie between 0 and 1'
        assert 0.0 <= split_valid_size <= 1.0, 'split_train_size does not lie between 0 and 1'
        assert split in ['train', 'valid', 'test']

        self.has_valid_set = (split_valid_size > 0.0)
        assert not (not self.has_valid_set and split == 'valid'), 'valid_size needs to be larger than 0'
        self.root = root
        self.split = split
        #self.data_file = os.path.join(self.root, data_set, 'data.txt')
        with open((os.path.join(self.root, 'crispr', data_set + '-torch-x.pkl')), 'rb') as f:
            x = torch.load(f)
        with open((os.path.join(self.root, 'crispr', data_set + '-torch-ymean.pkl')), 'rb') as f:
            y_mean = torch.load(f)
        with open((os.path.join(self.root, 'crispr', data_set + '-torch-yreplicates.pkl')), 'rb') as f:
            y_replicates = torch.load(f)
        with open((os.path.join(self.root, 'crispr', data_set + '-torch-sequence.pkl')), 'rb') as f:
            sequence = torch.load(f)

        d_vals = {}
        for i_v, val in enumerate(np.unique(x)):
            d_vals[val] = i_v
        for k, v in d_vals.items(): x[x == k] = v

        xoo_cols = []
        for col in range(x.shape[1]):
            xoo = np.zeros((x[:, col].size, x[:, col].max() + 1))
            xoo[np.arange(x[:, col].size), x[:, col]] = 1
            xoo_cols.append(xoo[:, :-1].astype(int))
        x_oo = np.concatenate(xoo_cols, axis=1)


        #xy_full = np.hstack([x, y_mean, y_replicates, np.reshape(sequence, (-1, 1))])
        #xy_full = np.hstack([x_oo, y_mean])

        # -
        if rep == 1:
            idxs_nonans = (np.isnan(y_replicates[:, 0]) == False)
            xy_full = np.hstack([x_oo[idxs_nonans, :], y_replicates[idxs_nonans,0].reshape((-1,1))])
        elif rep == 2:
            idxs_nonans = (np.isnan(y_replicates[:, 1]) == False)
            xy_full = np.hstack([x_oo[idxs_nonans, :], y_replicates[idxs_nonans,1].reshape((-1, 1))])
        elif rep == 3:
            idxs_nonans = (np.isnan(y_replicates[:, 2]) == False)
            xy_full = np.hstack([x_oo[idxs_nonans, :], y_replicates[idxs_nonans, 2].reshape((-1, 1))])
        else:
            #xy_full = np.hstack([x, y_mean, y_replicates, np.reshape(sequence, (-1, 1))])
            xy_full = np.hstack([x_oo, y_mean])

        # -


        xy_train, xy_test = modsel.train_test_split(
            xy_full, train_size=split_train_size, random_state=seed, shuffle=shuffle
        )
        if self.has_valid_set:
            xy_train, xy_valid = modsel.train_test_split(
                xy_train, train_size=1 - split_valid_size, random_state=seed, shuffle=shuffle
            )
            assert (len(xy_test) + len(xy_valid) + len(xy_train)) == len(xy_full)

        if scaling:
            self.scl = StandardScaler(copy=True)
            self.scl.fit(xy_train[:, :-1])
            xy_train[:, :-1] = self.scl.transform(xy_train[:, :-1])
            xy_test[:, :-1] = self.scl.transform(xy_test[:, :-1])
            self.m = xy_train[:, -1].mean()
            self.s = xy_train[:, -1].std()
            xy_train[:, -1] = (xy_train[:, -1] - self.m) / self.s
            xy_test[:, -1] = (xy_test[:, -1] - self.m) / self.s
            if self.has_valid_set:
                xy_valid[:, :-1] = self.scl.transform(xy_valid[:, :-1])
                xy_valid[:, -1] = (xy_valid[:, -1] - self.m) / self.s

        # impossible setting: if train is false, valid needs to be false too
        if split == 'train':
            self.data = torch.from_numpy(xy_train[:, :-1])
            self.targets = torch.from_numpy(xy_train[:, -1])
        elif split == 'valid':
            self.data = torch.from_numpy(xy_valid[:, :-1])
            self.targets = torch.from_numpy(xy_valid[:, -1])
        elif split == 'test':
            self.data = torch.from_numpy(xy_test[:, :-1])
            self.targets = torch.from_numpy(xy_test[:, -1])

        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()
        else:
            self.data = self.data.float()
            self.targets = self.targets.float()

        if self.targets.ndim == 1:
            # make (n_samples, 1) to comply with MSE
            self.targets = self.targets.unsqueeze(-1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    crispr = CrisprDatasets('flow-cytometry-HEK293')
    """
    from causa.utils import TensorDataLoader

    # NOTE: this is the setup without any validation set, otherwise decrease the test set size of the reamining data from 1.0 down to 0.5 e.g.
    ds_train = UCIRegressionDatasets('boston-housing', split_train_size=0.9, seed=711, root='data/', split='train',
                                     split_valid_size=0.0)
    ds_test = UCIRegressionDatasets('boston-housing', split_train_size=0.9, seed=711, root='data/', split='test',
                                    split_valid_size=0.0)
    # NOTE: use this for fast iterative data loader for small data sets like the one used here
    device = 'cpu'  # or 'cuda'
    train_loader = TensorDataLoader(ds_train.data.to(device), ds_train.targets.to(device), batch_size=len(ds_train))
    test_loader = TensorDataLoader(ds_test.data.to(device), ds_test.targets.to(device), batch_size=len(ds_train))
    for x, y in train_loader:
        print(x.shape, y.shape)
    """
