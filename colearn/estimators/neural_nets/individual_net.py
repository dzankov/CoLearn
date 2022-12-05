import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Sequential, Linear, ReLU, Softmax, Sigmoid
import torch_optimizer as optim
from sklearn.model_selection import train_test_split

from CIMtools.model_selection.transformation_out import TransformationOut


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MBSplitter(Dataset):

    def __init__(self, X, Y):
        super(MBSplitter, self).__init__()

        self.X = X
        self.Y = Y

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.Y)


class MainNet:
    def __new__(cls, ndim):
        inp_dim = ndim[0]

        net = []
        for dim in ndim[1:]:
            net.append(Linear(inp_dim, dim))
            net.append(ReLU())
            inp_dim = dim
        net.append(Linear(ndim[-1], 1))
        net = Sequential(*net)
        return net


class IndividualNet(nn.Module):

    def __init__(self, ndim=None, lmb=0, init_cuda=False):
        super().__init__()

        self.net = MainNet(ndim)
        self.lmb = lmb
        self.init_cuda = init_cuda


        if self.init_cuda:
            self.net.cuda()

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reset_weights(self):
        self.net.apply(self.reset_params)

    def train_val_split(self, X, Y, val_size=0.2, random_state=42, reacts=None):

        solvs = [i.meta['additive.1'] for i in reacts]
        solvs_num = {v: n for n, v in enumerate(set(solvs))}
        groups = [solvs_num[i] for i in solvs]
        rkf = TransformationOut(n_splits=5, n_repeats=1, shuffle=True, random_state=42)
        train, val = list(rkf.split(reacts, groups=groups))[0]

        X_train, X_val, Y_train, Y_val = X[train], X[val], Y[train], Y[val]

        #X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state=random_state)

        X_train, Y_train, = self.array_to_tensor(X_train, Y_train)
        X_val, Y_val, = self.array_to_tensor(X_val, Y_val)

        return X_train, X_val, Y_train, Y_val

    def get_batches(self, X, Y, batch_size=16):

        n_mb = np.ceil(len(X) / batch_size)
        batch_size = int(np.ceil(len(X) / n_mb))

        mb = DataLoader(MBSplitter(X, Y), batch_size=batch_size, shuffle=True)

        return mb

    def array_to_tensor(self, X, Y):

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X = torch.from_numpy(X.astype('float32'))
        Y = torch.from_numpy(Y.astype('float32'))

        if self.init_cuda:
            X, Y = X.cuda(), Y.cuda()

        return X, Y

    def loss(self, Y_pred, Y_true):

        mse = nn.MSELoss()
        loss = mse(Y_pred, Y_true)

        return loss

    def loss_batch(self, X_mb, Y_mb, optimizer=None):

        Y_out = self.forward(X_mb)

        total_loss = self.loss(Y_out, Y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def forward(self, X):
        out = self.net(X)
        return out

    def fit(self, X, Y, n_epoch=2000, batch_size=99999, lr=0.001, verbose=False, reacts=None):

        self.reset_weights()
        best_parameters = self.state_dict()

        X_train, X_val, Y_train, Y_val = self.train_val_split(X, Y, reacts=reacts)
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=self.lmb)

        val_loss = []
        for epoch in range(n_epoch):
            mb = self.get_batches(X_train, Y_train, batch_size=batch_size)
            self.train()
            for X_mb, Y_mb in mb:
                loss = self.loss_batch(X_mb, Y_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self.loss_batch(X_mb, Y_mb, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)
        return self

    def predict(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.forward(X)
        return np.asarray(pred.cpu())