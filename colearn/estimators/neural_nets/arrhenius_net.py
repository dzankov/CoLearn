import torch
import random
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

    def __init__(self, X, YK, YA, YE, T):
        super(MBSplitter, self).__init__()

        self.X = X
        self.YK = YK
        self.YA = YA
        self.YE = YE
        self.T = T

    def __getitem__(self, i):
        return self.X[i], self.YK[i], self.YA[i], self.YE[i], self.T[i]

    def __len__(self):
        return len(self.YK)


class MainNet(nn.Module):

    def __init__(self, ndim=None, out_dim=1, lmb=0):
        super().__init__()

        inp_dim = ndim[0]
        net = []
        for dim in ndim[1:]:
            net.append(Linear(inp_dim, dim))
            net.append(ReLU())
            inp_dim = dim
        net.append(Linear(ndim[-1], out_dim))

        self.net = Sequential(*net)
        self.optimizer = optim.Yogi(self.net.parameters(), weight_decay=lmb, lr=0.001)

    def update_weights(self, loss):

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reset_weights(self):
        self.net.apply(self.reset_params)

    def forward(self, X):
        out = self.net(X)
        return out


class ArrheniusConjugatedNet(nn.Module):

    def __init__(self, ndim=None, a=None, b=None, c=None, lmb=0, init_cuda=False):
        super().__init__()

        if a is None:
            self.a = torch.nn.Parameter(torch.Tensor([1.]))
        else:
            self.a = torch.Tensor([a])
        if b is None:
            self.b = torch.nn.Parameter(torch.Tensor([1.]))
        else:
            self.b = torch.Tensor([b])
        if c is None:
            self.c = torch.nn.Parameter(torch.Tensor([1.]))
        else:
            self.c = torch.Tensor([c])

        self.net = MainNet(ndim=ndim, out_dim=2, lmb=lmb)

        self.init_cuda = init_cuda

        if self.init_cuda:
            self.net.cuda()

    def reset_global_params(self):
        self.a = torch.nn.Parameter(torch.Tensor([1.]))
        self.b = torch.nn.Parameter(torch.Tensor([1.]))
        self.c = torch.nn.Parameter(torch.Tensor([1.]))
        return

    def train_val_split(self, X, YK, YA, YE, T, val_size=0.2, random_state=42, reacts=None):

        solvs = [i.meta['additive.1'] for i in reacts]
        solvs_num = {v: n for n, v in enumerate(set(solvs))}
        groups = [solvs_num[i] for i in solvs]
        rkf = TransformationOut(n_splits=5, n_repeats=1, shuffle=True, random_state=42)
        train, val = list(rkf.split(reacts, groups=groups))[0]

        (X_train, X_val, YK_train, YK_val,
         YA_train, YA_val, YE_train, YE_val,
         T_train, T_val) = (X[train], X[val], YK[train], YK[val],
                            YA[train], YA[val], YE[train], YE[val],
                            T[train], T[val])

        # (XK_train, XK_val, XA_train, XA_val,
        #  XE_train, XE_val, YK_train, YK_val,
        #  YA_train, YA_val, YE_train, YE_val,
        #  T_train, T_val) = train_test_split(XK, XA, XE, YK, YA, YE, T, test_size=val_size, random_state=random_state)

        (X_train, YK_train, YA_train, YE_train, T_train) = self.array_to_tensor(X_train, YK_train,
                                                                                YA_train, YE_train, T_train)

        X_val, YK_val, YA_val, YE_val, T_val = self.array_to_tensor(X_val, YK_val, YA_val, YE_val, T_val)

        return X_train, X_val, YK_train, YK_val, YA_train, YA_val, YE_train, YE_val, T_train, T_val

    def get_batches(self, X, YK, YA, YE, T, batch_size=16):

        n_mb = np.ceil(len(X) / batch_size)
        batch_size = int(np.ceil(len(X) / n_mb))

        mb = DataLoader(MBSplitter(X, YK, YA, YE, T), batch_size=batch_size, shuffle=True)

        return mb

    def array_to_tensor(self, X, YK, YA, YE, T):

        if YK.ndim == 1:
            YK = YK.reshape(-1, 1)
        if YA.ndim == 1:
            YA = YA.reshape(-1, 1)
        if YE.ndim == 1:
            YE = YE.reshape(-1, 1)
        if T.ndim == 1:
            T = T.reshape(-1, 1)

        X = torch.from_numpy(X.astype('float32'))
        YK = torch.from_numpy(YK.astype('float32'))
        YA = torch.from_numpy(YA.astype('float32'))
        YE = torch.from_numpy(YE.astype('float32'))

        T = torch.from_numpy(T.astype('float32'))

        if self.init_cuda:
            X, YK, YA, YE, T = X.cuda(), YK.cuda(), YA.cuda(), YE.cuda(), T.cuda()
        return X, YK, YA, YE, T

    def loss(self, YK_pred, YK_true, YA_pred, YA_true, YE_pred, YE_true):

        a = self.a.to(YK_pred.device)
        b = self.b.to(YA_pred.device)
        c = self.c.to(YE_pred.device)

        mse = nn.MSELoss()
        loss = a * mse(YK_pred, YK_true) + b * mse(YA_pred, YA_true) + c * mse(YE_pred, YE_true)

        return loss

    def loss_batch(self, X_mb, YK_mb, YA_mb, YE_mb, T_mb, optimizer=None):

        (XK_mb, XA_mb, XE_mb,
         YK_mb, YA_mb, YE_mb, T_mb) = (X_mb[torch.isfinite(YK_mb.flatten())],
                                       X_mb[torch.isfinite(YA_mb.flatten())],
                                       X_mb[torch.isfinite(YE_mb.flatten())],
                                       YK_mb[torch.isfinite(YK_mb.flatten())],
                                       YA_mb[torch.isfinite(YA_mb.flatten())],
                                       YE_mb[torch.isfinite(YE_mb.flatten())],
                                       T_mb[torch.isfinite(YK_mb.flatten())])

        YK_out = self.net.forward(XK_mb)[:, 0:1] - T_mb * self.net.forward(XK_mb)[:, 1:2]

        YA_out = self.net.forward(XA_mb)[:, 0:1]
        YE_out = self.net.forward(XE_mb)[:, 1:2]

        total_loss = self.loss(YK_out, YK_mb, YA_out, YA_mb, YE_out, YE_mb)
        if optimizer is not None:
            self.net.update_weights(total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def fit(self, X, YK, YA, YE, T, n_epoch=2000, batch_size=9999, lr=0.001, verbose=False, reacts=None):

        self.net.reset_weights()
        self.reset_global_params()

        (X_train, X_val, YK_train, YK_val,
         YA_train, YA_val, YE_train, YE_val,
         T_train, T_val) = self.train_val_split(X, YK, YA, YE, T, reacts=reacts)

        optimizer = optim.Yogi(self.parameters(), lr=lr)

        val_loss = []
        for epoch in range(n_epoch):
            mb = self.get_batches(X_train, YK_train, YA_train, YE_train, T_train, batch_size=batch_size)
            self.train()
            for X_mb, YK_mb, YA_mb, YE_mb, T_mb in mb:
                loss = self.loss_batch(X_mb, YK_mb, YA_mb, YE_mb, T_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self.loss_batch(X_mb, YK_mb, YA_mb, YE_mb, T_mb, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)

        return self

    def predict_YA(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.net.forward(X)[:, 0:1]
        return np.asarray(pred.cpu())

    def predict_YE(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.net.forward(X)[:, 1:2]
        return np.asarray(pred.cpu())

    def predict_YK(self, X, T):
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        T = T.astype('float32')
        pred = self.predict_YA(X) - T * self.predict_YE(X)
        return pred


class ArrheniusMTNet(ArrheniusConjugatedNet):

    def __init__(self, ndim=None, a=None, b=None, c=None, lmb=0, init_cuda=False):
        super().__init__(ndim=ndim, a=a, b=b, c=c, lmb=lmb, init_cuda=init_cuda)

        if a is None:
            self.a = torch.nn.Parameter(torch.Tensor([1.]))
        else:
            self.a = torch.Tensor([a])
        if b is None:
            self.b = torch.nn.Parameter(torch.Tensor([1.]))
        else:
            self.b = torch.Tensor([b])
        if c is None:
            self.c = torch.nn.Parameter(torch.Tensor([1.]))
        else:
            self.c = torch.Tensor([c])

        self.net = MainNet(ndim=ndim, out_dim=3, lmb=lmb)

        self.init_cuda = init_cuda

        if self.init_cuda:
            self.net.cuda()

    def loss_batch(self, X_mb, YK_mb, YA_mb, YE_mb, T_mb, optimizer=None):

        (XK_mb, XA_mb, XE_mb,
         YK_mb, YA_mb, YE_mb, T_mb) = (X_mb[torch.isfinite(YK_mb.flatten())],
                                       X_mb[torch.isfinite(YA_mb.flatten())],
                                       X_mb[torch.isfinite(YE_mb.flatten())],
                                       YK_mb[torch.isfinite(YK_mb.flatten())],
                                       YA_mb[torch.isfinite(YA_mb.flatten())],
                                       YE_mb[torch.isfinite(YE_mb.flatten())],
                                       T_mb[torch.isfinite(YK_mb.flatten())])

        YK_out = self.net.forward(XK_mb)[:, 0:1]
        YA_out = self.net.forward(XA_mb)[:, 1:2]
        YE_out = self.net.forward(XE_mb)[:, 2:3]

        total_loss = self.loss(YK_out, YK_mb, YA_out, YA_mb, YE_out, YE_mb)
        if optimizer is not None:
            self.net.update_weights(total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def predict_YA(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.net.forward(X)[:, 1:2]
        return np.asarray(pred.cpu())

    def predict_YE(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.net.forward(X)[:, 2:3]
        return np.asarray(pred.cpu())

    def predict_YK(self, X):

        X = torch.from_numpy(X.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                X = X.cuda()
            pred = self.net.forward(X)[:, 0:1]
        return np.asarray(pred.cpu())

