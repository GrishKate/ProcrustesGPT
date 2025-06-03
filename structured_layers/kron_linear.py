import torch
import torch.nn as nn
from procrustes.kron_utils import kron_project


def inverse_permutation(perm):
    inv = torch.empty(perm.shape, dtype=torch.int64)
    inv[perm] = torch.arange(perm.shape[0])
    return inv


class KronLinear(nn.Module):
    def __init__(self, r, m1, n1, m2, n2):
        super().__init__()
        A = torch.empty((r, m1, n1))
        B = torch.empty((r, m2, n2))
        nn.init.normal_(A, std=0.001)
        nn.init.normal_(B, std=0.001)

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = None
        self.m1 = m1
        self.n1 = n1
        self.m2 = m2
        self.n2 = n2
        self.r = r
        self.out_features = self.n1 * self.n2

    @property
    def weight(self):
        return torch.einsum('sia,sjb->abij', self.A, self.B).reshape(self.n1 * self.n2, self.m1 * self.m2)

    def assign(self, W, bias, dtype, inp_x_out=False, Qt=None, permute_bias=None):
        if inp_x_out:
            assert (self.A.shape == W[0].shape and self.B.shape == W[1].shape)
            del self.A, self.B
            self.A = nn.Parameter(W[0].to(dtype).cpu())
            self.B = nn.Parameter(W[1].to(dtype).cpu())
        else:
            assert (self.A.shape == W[0].transpose(1, 2).shape and self.B.shape == W[1].transpose(1, 2).shape)
            del self.A, self.B
            self.A = nn.Parameter(W[0].transpose(1, 2).to(dtype).cpu())
            self.B = nn.Parameter(W[1].transpose(1, 2).to(dtype).cpu())
        if self.bias is not None and bias is not None:
            self.bias = nn.Parameter(bias)
            if Qt is not None:
                self.bias.data = (Qt @ self.bias.data.to(Qt.device, torch.float32)).to(dtype).cpu()
            if permute_bias is not None:
                self.bias.data = self.bias.data[permute_bias]
            self.bias.data = self.bias.data.to(dtype).cpu()
        del W, bias, Qt

    @classmethod
    def get_from_layer(
            cls,
            layer,
            params,
            init=True,
            dtype=torch.float32
    ):
        r, m1, n1, m2, n2 = params['r'], params['m1'], params['n1'], params['m2'], params['n2']
        w, b = None, None
        if layer is not None:
            w, b = layer.weight, layer.bias
            if isinstance(layer, nn.Linear) or isinstance(layer, KronLinear):
                w = w.t()
            w = w.detach()
            assert w.shape[0] == m1 * m2
            assert w.shape[1] == n1 * n2
        kn_p_layer = cls(r, m1, n1, m2, n2)
        if b is not None:
            kn_p_layer.bias = b
        if init:
            assert w is not None
            kn_p_layer.initialize_AB(w, dtype)
        del w, b
        return kn_p_layer

    def initialize_AB(self, W, dtype):
        # algorithm 1 from https://arxiv.org/pdf/2401.16367
        # find A, B, p, q that minimize |pWq - A x_kron B |
        params = [{'m1': self.m1, 'm2': self.m2, 'n1': self.n1, 'n2': self.n2, 'r': self.r}]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        A, B = kron_project([W.to(device)], params)[0]
        del self.A, self.B
        self.A = nn.Parameter(A.cpu().to(dtype))
        self.B = nn.Parameter(B.cpu().to(dtype))
        del A, B, W

    def forward(self, x, make_full_matrix=True):
        if make_full_matrix:
            if len(x.shape) == 3:
                x = torch.einsum('nd,bsd->bsn', self.weight, x)
            else:
                x = torch.einsum('nd,bd->bn', self.weight, x)
        else:
            shape = x.shape
            x = x.view((-1, self.m1, self.m2))
            b = x.shape[0]
            # x=(b, m1, m2) B=(r, m2, n2) A=(r, m1, n1)
            if self.n2 + self.m1 > self.n1 + self.m2:
                B = self.B.transpose(0, 1).reshape(1, self.m2, -1).expand(
                    (x.shape[0], self.m2, self.r * self.n2))  # (b, m2, rn2)
                x = torch.bmm(x, B)  # (b, m1, rn2)
                x = x.reshape(b, self.m1, self.r, self.n2).permute(0, 3, 2, 1).reshape(b, self.n2, -1)  # (b, n2, rm1)
                A = self.A.reshape(1, self.r * self.m1, self.n1).expand((x.shape[0], self.r * self.m1, self.n1))
                x = torch.bmm(x, A)  # (b, n2, n1)
                x = x.transpose(1, 2).reshape(shape[:-1] + (self.n1 * self.n2,))
            else:
                A = self.A.transpose(0, 1).reshape(1, self.m1, -1).expand((x.shape[0], self.m1, self.r * self.n1))
                x = torch.bmm(x.transpose(1, 2), A)  # (b, m2, rn1)
                x = x.reshape(b, self.m2, self.r, self.n1).transpose(1, 3).reshape(b, self.n1, -1)  # (b, n1, rm2)
                B = self.B.reshape(1, self.r * self.m2, self.n2).expand(
                    (x.shape[0], self.r * self.m2, self.n2))  # (b, rm2, n2)
                x = torch.bmm(x, B)  # (b, n1, n1)
                x = x.reshape(shape[:-1] + (self.n1 * self.n2,))
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def get_params(self, inp_x_out=True):
        if not inp_x_out:
            return {'m1': self.n1, 'm2': self.n2, 'n1': self.m1, 'n2': self.m2, 'r': self.r}
        return {'m1': self.m1, 'm2': self.m2, 'n1': self.n1, 'n2': self.n2, 'r': self.r}
