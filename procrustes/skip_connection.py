import torch
import torch.nn as nn
from slicegpt.utils import cleanup_memory


class MatrixExp(nn.Module):
    def __init__(self, Q=None, S=None, sign=1, n=None, transpose=False, dtype=torch.float16):
        # Q is an orthogonal matrix, S is skew-symmetric
        # Q = torch.matrix_exp(S) = C exp(A) C^-1
        super().__init__()
        self.dtype = dtype
        self.transpose = transpose
        if Q is None:
            assert S is not None and sign is not None and n is not None
            self.S = S.to(dtype)
            self.sign = sign
            self.n = n
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Q = Q.to(device, torch.float32)
            self.n = Q.shape[0]
            sign, _ = torch.linalg.slogdet(Q)
            self.sign = 1 if sign > 0 else -1
            if self.sign == -1:
                if self.transpose:
                    Q[:, 0] = -Q[:, 0]
                else:
                    Q[0] = -Q[0]
            L, C = torch.linalg.eig(Q)
            S = torch.linalg.solve(C, C * torch.log(L)[None, :], left=False).real
            self.S = nn.Parameter(self.fold(S).to(dtype))
            del S, Q, L, C

    def fold(self, S):
        rows, cols = torch.triu_indices(S.size(0), S.size(1), offset=1)
        return S[rows, cols]

    def unfold(self):
        rows, cols = torch.triu_indices(self.n, self.n, offset=1, device=self.S.device)
        S = torch.zeros(self.n, self.n, device=self.S.device, dtype=self.S.dtype)
        S[rows, cols] = self.S
        S = S - S.T
        return S

    @property
    def weight(self):
        S = self.unfold()
        T = S
        Q = torch.eye(S.shape[0], device=S.device, dtype=S.dtype) + S
        for i in range(2, 10):
            T = (T @ S) / i
            Q += T
        if self.sign == -1:
            if self.transpose:
                Q[:, 0] = -Q[:, 0]
            else:
                Q[0] = -Q[0]
        return Q.to(self.dtype)

    def __matmul__(self, X):
        # return Q @ X
        return self.weight @ X

    def __rmatmul__(self, X):
        # return X @ Q
        return X @ self.weight

    @property
    def T(self):
        return MatrixExp(S=-self.S, sign=self.sign, n=self.n, transpose=True, dtype=self.dtype)

    def to(self, device):
        self.S.to(device)

    def forward(self, X):
        # return X @ Q
        return X @ self.weight


class Cayley(nn.Module):
    def __init__(self, Q=None, S=None, n=None, transpose=False, dtype=torch.float16):
        # Q is an orthogonal matrix, S is skew-symmetric
        super().__init__()
        self.dtype = dtype
        self.transpose = transpose
        self.U = None
        if Q is None:
            assert S is not None and n is not None
            self.S = S.to(dtype)
            self.n = n
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Q = Q.to(device=device, dtype=torch.float32)
            L, C = torch.linalg.eig(Q)
            inds = L.real < -1 + 1e-3
            if torch.sum(inds.to(torch.int32)) == 0:
                self.U = None
            else:
                U = C[:, inds].real
                inds = inds.nonzero()
                U = U / torch.norm(U, dim=0, keepdim=True)
                corr = U.T @ U
                X = []
                for i in range(corr.shape[0]):
                    flag = True
                    for j in range(i + 1, corr.shape[0]):
                        if abs(corr[i][j] - 1) < 1e-4:
                            flag = False
                    if flag:
                        X.append(U[:, i])
                    else:
                        X.append(C[:, inds[i]].imag.squeeze())
                self.U = torch.stack(X, dim=-1)
                self.U = self.U / torch.norm(self.U, dim=0, keepdim=True)
                self.U = self.U.unsqueeze(2)
                for i in range(self.U.shape[1]):
                    Q = Q - 2 * self.U[:, i] @ (self.U[:, i].T @ Q)  # multiply by Householder matrix
                del X, U
            del L, C
            self.n = Q.shape[0]
            I = torch.eye(self.n, device=Q.device)
            S = torch.linalg.solve(I + Q, I - Q, left=self.transpose)
            self.S = nn.Parameter(self.fold(S).to(dtype))
            del I, S, Q

    def fold(self, S):
        rows, cols = torch.triu_indices(S.size(0), S.size(1), offset=0)
        return S[rows, cols]

    def unfold(self):
        rows, cols = torch.triu_indices(self.n, self.n, offset=0, device=self.S.device)
        S = torch.zeros(self.n, self.n, device=self.S.device, dtype=self.S.dtype)
        S[rows, cols] = self.S
        S = S - S.T
        return S

    @property
    def weight(self):
        S = self.unfold()
        I = torch.eye(S.shape[0], device=S.device)
        Q = torch.linalg.solve(I + S, I - S, left=not self.transpose)
        if self.U is not None:
            for i in range(self.U.shape[1] - 1, -1, -1):
                Q = Q - 2 * self.U[:, i] @ (self.U[:, i].T @ Q)
        return Q.to(self.dtype)

    def __matmul__(self, X):
        # return Q @ X
        return self.weight @ X

    def __rmatmul__(self, X):
        # return X @ Q
        return X @ self.weight

    @property
    def T(self):
        return Cayley(S=-self.S, n=self.n, transpose=True, dtype=self.dtype)

    def to(self, device):
        self.S.to(device)

    def forward(self, X):
        # return X @ Q
        return X @ self.weight


def replace_one_skip(items):
    torch.set_num_threads(1)
    i, layer_adapter, cls = items
    Qskip = layer_adapter.layer.attn_shortcut_Q
    del layer_adapter.layer.attn_shortcut_Q
    layer_adapter.layer.attn_shortcut_Q = cls(Qskip)
    # print(i, type(layer_adapter.layer.attn_shortcut_Q),
    #      torch.norm(Qskip - layer_adapter.layer.attn_shortcut_Q.weight.to(Qskip.device, Qskip.dtype)).item())
    del Qskip

    Qskip = layer_adapter.layer.mlp_shortcut_Q
    del layer_adapter.layer.mlp_shortcut_Q
    layer_adapter.layer.mlp_shortcut_Q = cls(Qskip)
    # print(i, type(layer_adapter.layer.mlp_shortcut_Q),
    #      torch.norm(Qskip - layer_adapter.layer.mlp_shortcut_Q.weight.to(Qskip.device, Qskip.dtype)).item())
    del Qskip


def process_skip_connections(model_adapter, skip_con_decomp, n_process=1):
    if skip_con_decomp == 'exponent':
        cls = MatrixExp
    elif skip_con_decomp == 'cayley':
        cls = Cayley
    else:
        raise NotImplementedError
    layers = model_adapter.get_layers()
    lst = [(i, layers[i], cls) for i in range(len(layers))]
    for item in lst:
        replace_one_skip(item)
    cleanup_memory()
