import torch
import torch.nn as nn
from procrustes.kron_utils import kron_project


class KronEmbedding(nn.Module):
    def __init__(self, r, m1, n1, m2, n2):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.m1 = m1
        self.m2 = m2
        self.r = r
        self.A = nn.Parameter(torch.randn(r, m1, n1))
        self.B = nn.Parameter(torch.randn(r, m2, n2))
        self.out_features = self.n1 * self.n2

    @classmethod
    def get_from_layer(
            cls,
            layer,
            params,
            init=False,
            dtype=torch.float32
    ):
        assert isinstance(layer, nn.Embedding)
        r, m1, n1, m2, n2 = params['r'], params['m1'], params['n1'], params['m2'], params['n2']
        w = layer.weight.detach()
        vocab_size = w.shape[0]
        emb = w.shape[1]

        assert vocab_size == m1 * m2
        assert emb == n1 * n2

        kn_p_layer = cls(r, m1, n1, m2, n2)
        if init:
            kn_p_layer.initialize_AB(w, dtype=dtype)
        del w
        return kn_p_layer

    def initialize_AB(self, W, dtype=torch.float32):
        params = [{'m1': self.m1, 'm2': self.m2, 'n1': self.n1, 'n2': self.n2, 'r': self.r}]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        A, B = kron_project([W.to(device)], params)[0]
        del self.A, self.B
        self.A = nn.Parameter(A.cpu().to(dtype))
        self.B = nn.Parameter(B.cpu().to(dtype))
        del A, B, W

    @property
    def weight(self):
        return torch.einsum('sia,sjb->ijab', self.A, self.B).reshape(self.m1 * self.m2, self.n1 * self.n2)

    def forward(self, x):
        return self.weight[x]

    def assign(self, W, dtype, inp_x_out=False):
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

    def get_params(self, inp_x_out=True):
        if not inp_x_out:
            return {'m1': self.n1, 'm2': self.n2, 'n1': self.m1, 'n2': self.m2, 'r': self.r}
        return {'m1': self.m1, 'm2': self.m2, 'n1': self.n1, 'n2': self.n2, 'r': self.r}


class KronOPTLearnedPositionalEmbedding(KronEmbedding):

    def __init__(self, r, m1, n1, m2, n2):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(r, m1, n1, m2, n2)

    def forward(
            self,
            attention_mask: torch.LongTensor,
            past_key_values_length: int = 0,
            position_ids=None,
    ):
        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_key_values_length` is > 0
            position_ids = position_ids[:, past_key_values_length:]

        return super().forward(position_ids + self.offset)
