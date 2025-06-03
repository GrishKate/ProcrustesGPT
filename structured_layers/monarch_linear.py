import torch
import torch.nn as nn
from einops import rearrange
from procrustes.monarch_utils import gs_project_matrix, inverse_permutation, generate_perfect_shuffle, form_full


class MonarchLinear(nn.Module):
    def __init__(self, kl, bl1, bl2, kr, br1, br2, use_pl=True, use_pr=True):
        super().__init__()
        self.kl, self.bl1, self.bl2 = kl, bl1, bl2
        self.kr, self.br1, self.br2 = kr, br1, br2
        L = torch.empty((kl, bl1, bl2))
        R = torch.empty((kr, br1, br2))
        self.L = nn.Parameter(L)
        self.R = nn.Parameter(R)
        self.bias = None
        self.register_buffer("p", torch.tensor(generate_perfect_shuffle(kl, bl2 * kl)))

        self.use_pl = use_pl
        self.use_pr = use_pr
        if self.use_pl:
            self.register_buffer("pl", torch.tensor(inverse_permutation(generate_perfect_shuffle(kl, bl1 * kl))))
        else:
            self.pl = None
        if self.use_pr:
            self.register_buffer("pr", torch.tensor(inverse_permutation(generate_perfect_shuffle(kr, br2 * kr))))
        else:
            self.pr = None
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def weight(self):
        return form_full(self.L, self.R, self.pl, self.p, self.pr)

    def assign(self, W, bias, dtype, inp_x_out=True, Qt=None, permute_bias=None, device=torch.device('cpu')):
        if inp_x_out:
            assert (self.L.shape == W[0].shape and self.R.shape == W[1].shape)
            L, R, pl, p, pr = W
            self.L = nn.Parameter(L.to(dtype).to(device))
            self.R = nn.Parameter(R.to(dtype).to(device))
            self.pl = torch.tensor(pl, dtype=torch.long) if pl is not None else None
            self.p = torch.tensor(p, dtype=torch.long) if p is not None else None
            self.pr = torch.tensor(pr, dtype=torch.long) if pr is not None else None
            del L, R, pl, p, pr
        else:
            raise NotImplementedError
            assert (self.L.shape == W[0].transpose(1, 2).shape and self.R.shape == W[1].transpose(1, 2).shape)
            self.L = nn.Parameter(W[0].transpose(1, 2).to(dtype).to(device))
            self.R = nn.Parameter(W[1].transpose(1, 2).to(dtype).to(device))
        if self.bias is not None and bias is not None:
            self.bias.data = bias.data.to(dtype).cpu()
            if Qt is not None:
                self.bias.data = (Qt @ self.bias.data.to(Qt.device, torch.float32)).to(dtype).to(device)
            if permute_bias is not None:
                self.bias.data = self.bias.data[permute_bias]
        del W, bias, Qt

    @classmethod
    def get_from_layer(
            cls,
            layer,
            params,
            init=True,
            dtype=torch.float32
    ):
        kl, bl1, bl2 = params['kl'], params['bl1'], params['bl2']
        kr, br1, br2 = params['kr'], params['br1'], params['br2']
        use_pl, use_pr = params['use_pl'], params['use_pr']
        w, b = None, None
        if layer is not None:
            w, b = layer.weight, layer.bias
            if isinstance(layer, nn.Linear):
                w = w.t()
            w = w.detach()
            assert w.shape[0] == kl * bl1
            assert w.shape[1] == kr * br2
        new_layer = cls(kl, bl1, bl2, kr, br1, br2, use_pl=use_pl, use_pr=use_pr)
        if b is not None:
            new_layer.bias = b
        if init:
            assert w is not None
            new_layer.initialize_weights(w, dtype)
        del w, b
        return new_layer

    def initialize_weights(self, W, dtype):
        params = {'kl': self.kl, 'bl1': self.bl1, 'bl2': self.bl2, 'kr': self.kr, 'br1': self.br1,
                  'br2': self.br2, 'use_pl': self.use_pl, 'use_pr': self.use_pr}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        L, R, pl, p, pr = gs_project_matrix(W.to(device), params, dtype, inp_x_out=True)
        del self.L, self.R, self.pl, self.p, self.pr
        self.L = nn.Parameter(L.cpu().to(dtype))
        self.R = nn.Parameter(R.cpu().to(dtype))
        self.pl = torch.tensor(pl, dtype=torch.long) if pl is not None else None
        self.p = torch.tensor(p, dtype=torch.long) if p is not None else None
        self.pr = torch.tensor(pr, dtype=torch.long) if pr is not None else None
        del W, L, R, pl, p, pr

    def forward(self, X):
        """
        Returns batched product X @ (pl L p R pr)
        Input:
        X (..., kr*br1)
        L (kl, bl1, bl2)
        R (kr, br1, br2)
        Output:
        Y (...,kl * bl1)
        """
        batch_shape = X.shape[:-1]
        m = X.shape[-1]
        X = X.view(-1, m)
        kl, bl1, bl2 = self.L.shape
        kr, br1, br2 = self.R.shape
        assert kl * bl2 == kr * br1
        """
        full_M = torch.block_diag(*[self.L[i] for i in range(len(self.L))])
        full_M = full_M[:, self.p] @ torch.block_diag(*[self.R[i] for i in range(len(self.R))])
        if self.pl is not None:
            full_M = full_M[self.pl]
        if self.pr is not None:
            full_M = full_M[:, self.pr]
        Y = X @ full_M
        """


        if self.data is not None:
            Y = X @ self.data

        if self.pl is not None:
            X = X[..., inverse_permutation(self.pl)]

        X = rearrange(X, 'b (k j) -> k b j', k=kl)  # (kl, batch, bl1)
        X = torch.bmm(X, self.L)  # (kl, batch, bl2)

        X = rearrange(X, 'k b j -> b (k j)')[..., self.p]
        X = rearrange(X, 'b (k i) -> k b i', k=kr)  # (kr, batch, br1)

        X = torch.bmm(X, self.R)  # (kr, batch, br2)
        X = rearrange(X, 'k b i -> b (k i)')
        if self.pr is not None:
            X = X[..., self.pr]
        # print(torch.norm(X-Y))
        if self.data is not None:
            print('here', torch.norm(Y - X).item(), torch.norm(self.data - self.weight).item())

        if self.bias is not None:
            return X.reshape(*batch_shape, X.shape[-1]) + self.bias
        else:
            return X.reshape(*batch_shape, X.shape[-1])

    @property
    def out_features(self):
        return self.kr * self.br2

    @property
    def T(self):
        Q = MonarchLinear(self.kr, self.br2, self.br1, self.kl, self.bl2, self.bl1, self.use_pr, self.use_pl)
        Q.assign([self.R.mT, self.L.mT, self.pr, inverse_permutation(self.p), self.pl], bias=None, dtype=torch.float32, inp_x_out=True,
        device=self.L.device)
        return Q


    def __rmatmul__(self, X):
        # return X @ Q
        return self.forward(X)

    def __matmul__(self, X):
        """
        full_M = torch.block_diag(*[self.L[i] for i in range(len(self.L))])
        full_M = full_M[:, self.p] @ torch.block_diag(*[self.R[i] for i in range(len(self.R))])
        if self.pl is not None:
            full_M = full_M[self.pl]
        if self.pr is not None:
            full_M = full_M[:, self.pr]
        Y = full_M @ X
        """

        # return Q @ X
        #batch_shape = X.shape[:-1]
        #m = X.shape[-1]
        #X = X.view(-1, m)
        kl, bl1, bl2 = self.L.shape
        kr, br1, br2 = self.R.shape
        assert kl * bl2 == kr * br1
        shape = X.shape
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)

        if self.pr is not None:
            X = X[inverse_permutation(self.pr), ...]

        X = rearrange(X, '(k j) b -> k j b', k=kr)  # (kr, br2, batch)
        X = torch.bmm(self.R, X)  # (kr, br1, batch)

        X = rearrange(X, 'k j b -> (k j) b')[inverse_permutation(self.p), ...]
        X = rearrange(X, '(k i) b -> k i b', k=kl)  # (kl, bl2, batch)

        X = torch.bmm(self.L, X)  # (kl, bl1, batch)
        X = rearrange(X, 'k i b -> (k i) b')
        if self.pl is not None:
            X = X[self.pl, ...]
        #print('result', X.shape)
        #print(torch.norm(X-Y).item())
        return X.reshape(shape)
