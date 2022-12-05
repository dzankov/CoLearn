import torch
import numpy as np


class IndividualRidge:

    def __init__(self, lmb=0, init_cuda=False):
        self.lmb = lmb
        self.init_cuda = init_cuda

    def _add_bias(self, X):
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        return X_new

    def array_to_tensor(self, X, Y):

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X = torch.from_numpy(self._add_bias(X.astype('float64')))
        Y = torch.from_numpy(Y.astype('float64'))
        I = torch.from_numpy(np.eye(X.shape[1]))

        if self.init_cuda:
            X, Y, I = X.cuda(), Y.cuda(), I.cuda()
        return X, Y, I

    def fit(self, X, Y):
        X, Y, I = self.array_to_tensor(X, Y)
        self.W_ = torch.inverse(X.transpose(0, 1).mm(X) + self.lmb * I).mm(X.transpose(0, 1)).mm(Y)
        return self

    def predict(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.W_).cpu()
        return pred


class TautomerismConjugatedRidge:

    def __init__(self, alpha=0, lmb=0, init_cuda=False):

        self.alpha = alpha
        self.lmb = lmb
        self.init_cuda = init_cuda

    def _add_bias(self, X):

        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X

        return X_new

    def array_to_tensor(self, X1, X2, X, YT, YA):

        if YT.ndim == 1:
            YT = YT.reshape(-1, 1)
        if YA.ndim == 1:
            YA = YA.reshape(-1, 1)

        X1 = torch.from_numpy(self._add_bias(X1.astype('float64')))
        X2 = torch.from_numpy(self._add_bias(X2.astype('float64')))
        X = torch.from_numpy(self._add_bias(X.astype('float64')))

        YT = torch.from_numpy(YT.astype('float64'))
        YA = torch.from_numpy(YA.astype('float64'))

        I = torch.from_numpy(np.eye(X.shape[1]))

        if self.init_cuda:
            X1, X2, X, YT, YA, I = X1.cuda(), X2.cuda(), X.cuda(), YT.cuda(), YA.cuda(), I.cuda()
        return X1, X2, X, YT, YA, I

    def fit(self, X1, X2, X, YT, YA):

        X1, X2, X, YT, YA, I = self.array_to_tensor(X1, X2, X, YT, YA)

        z1 = torch.inverse(self.alpha * (X2 - X1).transpose(0, 1).mm(X2 - X1) + (1 - self.alpha) * X.transpose(0, 1).mm(X) + self.lmb * I)

        z2 = (self.alpha * (X2 - X1).transpose(0, 1).mm(YT) + (1 - self.alpha) * X.transpose(0, 1).mm(YA))

        self.W_ = z1.mm(z2)

        return self

    def predict_acidity(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.W_).cpu()
        return pred

    def predict_constant(self, X1, X2):
        pred = self.predict_acidity(X2) - self.predict_acidity(X1)
        return pred


class ArrheniusConjugatedRidge:

    def __init__(self, a=1, b=1, c=1, lmbA=1, lmbE=1, init_cuda=False):

        self.a = a
        self.b = b
        self.c = c
        self.lmbA = lmbA
        self.lmbE = lmbE
        self.init_cuda = init_cuda

    def _add_bias(self, X):
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        return X_new

    def array_to_tensor(self, XK, XA, XE, YK, YA, YE, T):

        if YK.ndim == 1:
            YK = YK.reshape(-1, 1)
        if YA.ndim == 1:
            YA = YA.reshape(-1, 1)
        if YE.ndim == 1:
            YE = YE.reshape(-1, 1)

        XK = self._add_bias(XK)
        XA = self._add_bias(XA)
        XE = self._add_bias(XE)
        T = np.diag(T)

        XK = torch.from_numpy(XK.astype('float64'))
        XA = torch.from_numpy(XA.astype('float64'))
        XE = torch.from_numpy(XE.astype('float64'))

        YK = torch.from_numpy(YK.astype('float64'))
        YA = torch.from_numpy(YA.astype('float64'))
        YE = torch.from_numpy(YE.astype('float64'))

        T = torch.from_numpy(T.astype('float64'))
        I = torch.from_numpy(np.eye(XK.shape[1]).astype('float64'))

        if self.init_cuda:
            XK, XA, XE, YK, YA, YE, T, I = XK.cuda(), XA.cuda(), XE.cuda(), YK.cuda(), YA.cuda(), YE.cuda(), T.cuda(), I.cuda()
        return XK, XA, XE, YK, YA, YE, T, I

    def fit(self, XK, XA, XE, YK, YA, YE, T):

        XK, XA, XE, YK, YA, YE, T, I = self.array_to_tensor(XK, XA, XE, YK, YA, YE, T)

        a1 = torch.inverse(self.a * XK.transpose(0, 1).mm(XK) + self.b * XA.transpose(0, 1).mm(XA) + self.lmbA * I)
        a2 = self.a * XK.transpose(0, 1).mm(YK) + self.b * XA.transpose(0, 1).mm(YA)
        a3 = self.a * XK.transpose(0, 1).mm(T).mm(XK)

        b1 = torch.inverse(self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(T).mm(XK) + self.c * XE.transpose(0, 1).mm(XE) + self.lmbE * I)
        b2 = self.c * XE.transpose(0, 1).mm(YE) - self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(YK)
        b3 = self.a * XK.transpose(0, 1).mm(T.transpose(0, 1)).mm(XK)

        A, B, C, D = b1.mm(b2), b1.mm(b3), a1.mm(a2), a1.mm(a3)

        self.WA_ = torch.inverse(I - D.mm(B)).mm(C + D.mm(A))
        self.WE_ = torch.inverse(I - B.mm(D)).mm(A + B.mm(C))

        return self

    def predict_YA(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WA_).cpu()
        return pred

    def predict_YE(self, X):

        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WE_).cpu()
        return pred

    def predict_YK(self, X, T):
        T = torch.from_numpy(np.diag(T).astype('float64'))
        pred = self.predict_YA(X) - T.mm(self.predict_YE(X))
        return pred


class ArrheniusMTRidge(ArrheniusConjugatedRidge):

    def __init__(self, a=1, b=1, c=1, lmbK=1, lmbA=1, lmbE=1, init_cuda=False):
        super().__init__(a=a, b=b, c=c, lmbA=lmbA, lmbE=lmbE, init_cuda=init_cuda)
        self.lmbK = lmbK


    def fit(self, XK, XA, XE, YK, YA, YE, T):

        XK, XA, XE, YK, YA, YE, T, I = self.array_to_tensor(XK, XA, XE, YK, YA, YE, T)

        self.WK_ = torch.inverse(self.a * XK.transpose(0, 1).mm(XK) + self.lmbK * I).mm(self.a * XK.transpose(0, 1)).mm(YK)
        self.WA_ = torch.inverse(self.b * XA.transpose(0, 1).mm(XA) + self.lmbA * I).mm(self.b * XA.transpose(0, 1)).mm(YA)
        self.WE_ = torch.inverse(self.c * XE.transpose(0, 1).mm(XE) + self.lmbE * I).mm(self.c * XE.transpose(0, 1)).mm(YE)

        return self

    def predict_YK(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WK_).cpu()
        return pred


class SelectivityConjugatedRidge:

    def __init__(self, a=1, b=1, c=1, lmbE=1, lmbS=1, init_cuda=False):

        self.a = a
        self.b = b
        self.c = c
        self.lmbE = lmbE
        self.lmbS = lmbS
        self.init_cuda = init_cuda

    def _add_bias(self, X):
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        return X_new

    def array_to_tensor(self, XE, XS, XEP, XSP, YE, YS, YP):

        if YP.ndim == 1:
            YP = YP.reshape(-1, 1)
        if YE.ndim == 1:
            YE = YE.reshape(-1, 1)
        if YS.ndim == 1:
            YS = YS.reshape(-1, 1)

        XE = self._add_bias(XE)
        XS = self._add_bias(XS)
        XEP = self._add_bias(XEP)
        XSP = self._add_bias(XSP)

        XE = torch.from_numpy(XE.astype('float64'))
        XS = torch.from_numpy(XS.astype('float64'))
        XEP = torch.from_numpy(XEP.astype('float64'))
        XSP = torch.from_numpy(XSP.astype('float64'))

        YE = torch.from_numpy(YE.astype('float64'))
        YS = torch.from_numpy(YS.astype('float64'))
        YP = torch.from_numpy(YP.astype('float64'))

        I = torch.from_numpy(np.eye(XE.shape[1]).astype('float64'))

        if self.init_cuda:
            XE, XS, XEP, XSP, YE, YS, YP, I = XE.cuda(), XS.cuda(), XEP.cuda(), XSP.cuda(), YE.cuda(), YS.cuda(), YP.cuda(), I.cuda()
        return XE, XS, XEP, XSP, YE, YS, YP, I

    def fit(self, XE, XS, XEP, XSP, YE, YS, YP):

        XE, XS, XEP, XSP, YE, YS, YP, I = self.array_to_tensor(XE, XS, XEP, XSP, YE, YS, YP)

        a1 = torch.inverse(self.c * XEP.transpose(0, 1).mm(XEP) + self.a * XE.transpose(0, 1).mm(XE) + self.lmbE * I)
        a2 = (self.a * XE.transpose(0, 1).mm(YE) + self.c * XEP.transpose(0, 1).mm(YP))
        a3 = self.c * XEP.transpose(0, 1).mm(XSP)

        b1 = torch.inverse(self.c * XSP.transpose(0, 1).mm(XSP) + self.b * XS.transpose(0, 1).mm(XS) + self.lmbS * I)
        b2 = (self.b * XS.transpose(0, 1).mm(YS) - self.c * XSP.transpose(0, 1).mm(YP))
        b3 = self.c * XSP.transpose(0, 1).mm(XEP)

        A, B, C, D = a1.mm(a2), a1.mm(a3), b1.mm(b2), b1.mm(b3)

        self.WE_ = torch.inverse(I - B.mm(D)).mm(A + B.mm(C))
        self.WS_ = torch.inverse(I - D.mm(B)).mm(C + D.mm(A))

        return self

    def predict_YE(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WE_).cpu()
        return pred

    def predict_YS(self, X):
        X = torch.from_numpy(self._add_bias(X).astype('float64'))
        if self.init_cuda:
            X = X.cuda()
        pred = X.mm(self.WS_).cpu()
        return pred

    def predict_YP(self, XE, XS):
        pred = self.predict_YE(XE) - self.predict_YS(XS)
        return pred


class ArrheniusIndividualRidge:

    def __init__(self, lmbA=0, lmbE=0, init_cuda=False):

        self.lmbA = lmbA
        self.lmbE = lmbE
        self.init_cuda = init_cuda

    def fit(self, XA, XE, YA, YE):

        self.ridge_YA = IndividualRidge(lmb=self.lmbA, init_cuda=self.init_cuda)
        self.ridge_YA.fit(XA, YA)

        self.ridge_YE = IndividualRidge(lmb=self.lmbE, init_cuda=self.init_cuda)
        self.ridge_YE.fit(XE, YE)

        return self

    def predict_YA(self, X):
        pred = self.ridge_YA.predict(X)
        return pred

    def predict_YE(self, X):
        pred = self.ridge_YE.predict(X)
        return pred

    def predict_YK(self, X, T):
        T = torch.from_numpy(T.astype('float64'))
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        pred = self.predict_YA(X) - T * self.predict_YE(X)
        return pred



