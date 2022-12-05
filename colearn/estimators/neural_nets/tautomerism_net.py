import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Sequential, Linear, ReLU, Softmax, Sigmoid
import torch_optimizer as optim
from sklearn.model_selection import train_test_split


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


class TauMBSplitter(Dataset):

    def __init__(self, X1, X2, Y):
        super(TauMBSplitter, self).__init__()

        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.Y[i]

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


class TautomerismConjugatedNet(nn.Module):

    def __init__(self, ndim=None, alpha=1, init_cuda=False):
        super().__init__()

        self.alpha = alpha
        self.net = MainNet(ndim)
        self.init_cuda = init_cuda

        if self.init_cuda:
            self.net.cuda()

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reset_weights(self):
        self.net.apply(self.reset_params)

    def train_val_split(self, X1, X2, X, YT, YA, val_size=0.2, random_state=42):

        X1_train, X1_val, X2_train, X2_val, YT_train, YT_val = train_test_split(X1, X2, YT, test_size=val_size, random_state=random_state)

        X_train, X_val, YA_train, YA_val = train_test_split(X, YA, test_size=val_size, random_state=random_state)

        X1_train, X2_train, X_train, YT_train, YA_train = self.array_to_tensor(X1_train, X2_train, X_train, YT_train, YA_train)
        X1_val, X2_val, X_val, YT_val, YA_val = self.array_to_tensor(X1_val, X2_val, X_val, YT_val, YA_val)

        return X1_train, X1_val, X2_train, X2_val, X_train, X_val, YT_train, YT_val, YA_train, YA_val

    def get_batches(self, X1, X2, X, YT, YA, batch_size=16):

        n_mb = np.ceil(len(X1) / batch_size)
        batch_size = int(np.ceil(len(X) / n_mb))

        T_mb = DataLoader(TauMBSplitter(X1, X2, YT), batch_size=batch_size, shuffle=True)
        A_mb = DataLoader(MBSplitter(X, YA), batch_size=batch_size, shuffle=True)

        return T_mb, A_mb

    def array_to_tensor(self, X1, X2, X, YT, YA):

        if YT.ndim == 1:
            YT = YT.reshape(-1, 1)
        if YA.ndim == 1:
            YA = YA.reshape(-1, 1)

        X1 = torch.from_numpy(X1.astype('float32'))
        X2 = torch.from_numpy(X2.astype('float32'))
        X = torch.from_numpy(X.astype('float32'))

        YT = torch.from_numpy(YT.astype('float32'))
        YA = torch.from_numpy(YA.astype('float32'))

        if self.init_cuda:
            X1, X2, X, YT, YA = X1.cuda(), X2.cuda(), X.cuda(), YT.cuda(), YA.cuda()

        return X1, X2, X, YT, YA

    def loss(self, YT_pred, YT_true, YA_pred, YA_true):

        mse = nn.MSELoss()
        T_loss = mse(YT_pred, YT_true)
        A_loss = mse(YA_pred, YA_true)
        loss = self.alpha * T_loss + (1 - self.alpha) * A_loss

        return loss

    def loss_batch(self, X1_mb, X2_mb, X_mb, YT_mb, YA_mb, optimizer=None):

        YT_out = self.forward(X2_mb) - self.forward(X1_mb)
        YA_out = self.forward(X_mb)

        total_loss = self.loss(YT_out, YT_mb, YA_out, YA_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def forward(self, X):
        out = self.net(X)
        return out

    def fit(self, X1, X2, X, YT, YA, n_epoch=1000, batch_size=9999, lr=0.001, weight_decay=0.1, verbose=False):

        self.reset_weights()

        X1_train, X1_val, X2_train, X2_val, X_train, X_val, YT_train, YT_val, YA_train, YA_val = self.train_val_split(X1, X2, X, YT, YA)
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=weight_decay)

        val_loss = []
        for epoch in range(n_epoch):
            T_mb, A_mb = self.get_batches(X1_train, X2_train, X_train, YT_train, YA_train, batch_size=batch_size)
            self.train()
            for (X1_mb, X2_mb, YT_mb), (X_mb, YA_mb) in zip(T_mb, A_mb):
                loss = self.loss_batch(X1_mb, X2_mb, X_mb, YT_mb, YA_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self.loss_batch(X1_mb, X2_mb, X_mb, YT_mb, YA_mb, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)
        return self

    def predict_acidity(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.forward(X)
        return np.asarray(pred.cpu())

    def predict_constant(self, X1, X2):
        pred = self.predict_acidity(X2) - self.predict_acidity(X1)
        return pred