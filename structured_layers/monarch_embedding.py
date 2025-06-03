import torch
import torch.nn as nn
from procrustes.monarch_utils import form_full, gs_project_matrix, inverse_permutation


def generate_perfect_shuffle(k, n):
    p = []
    for i in range(n):
        p.append((i % k) * (n // k) + i // k)
    return p


class MonarchEmbedding(nn.Module):
    def __init__(self, kl, bl1, bl2, kr, br1, br2, use_pl=False, use_pr=False):
        super().__init__()
        self.kl, self.bl1, self.bl2 = kl, bl1, bl2
        self.kr, self.br1, self.br2 = kr, br1, br2
        L = torch.empty((kl, bl1, bl2))
        R = torch.empty((kr, br1, br2))
        nn.init.normal_(L, std=0.001)
        nn.init.normal_(R, std=0.001)
        self.L = nn.Parameter(L)
        self.R = nn.Parameter(R)
        self.register_buffer("p", torch.tensor(generate_perfect_shuffle(kl, bl2 * kl)))

        self.use_pl = use_pl
        self.use_pr = use_pr
        if self.use_pl:
            self.register_buffer("pl", torch.tensor(inverse_permutation(torch.tensor(generate_perfect_shuffle(kl, bl1 * kl)))))
        else:
            self.pl = None
        if self.use_pr:
            self.register_buffer("pr", torch.tensor(inverse_permutation(torch.tensor(generate_perfect_shuffle(kr, br2 * kr)))))
        else:
            self.pr = None

    @classmethod
    def get_from_layer(
            cls,
            layer,
            params,
            init=False,
            dtype=torch.float32
    ):
        assert isinstance(layer, nn.Embedding)
        kl, bl1, bl2 = params['kl'], params['bl1'], params['bl2']
        kr, br1, br2 = params['kr'], params['br1'], params['br2']
        use_pl, use_pr = params['use_pl'], params['use_pr']
        w = layer.weight.detach()
        vocab_size = w.shape[0]
        emb = w.shape[1]

        assert vocab_size == kl * bl1
        assert emb == kr * br2

        new_layer = cls(kl, bl1, bl2, kr, br1, br2, use_pl=use_pl, use_pr=use_pr)
        if init:
            new_layer.initialize_weights(w, dtype=dtype)
        del w
        return new_layer

    def initialize_weights(self, W, dtype=torch.float32):
        params = {'kl': self.kl, 'bl1': self.bl1, 'bl2': self.bl2, 'kr': self.kr, 'br1': self.br1, 'br2': self.br2,
                  'use_pl': self.use_pl, 'use_pr': self.use_pr}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        L, R, pl, p, pr = gs_project_matrix(W.to(device, torch.float32), params, dtype, inp_x_out=True)
        del self.L, self.R, self.pl, self.p, self.pr
        self.L = nn.Parameter(L.cpu().to(dtype))
        self.R = nn.Parameter(R.cpu().to(dtype))
        self.pl = torch.tensor(pl, dtype=torch.long) if pl is not None else None
        self.p = torch.tensor(p, dtype=torch.long) if p is not None else None
        self.pr = torch.tensor(pr, dtype=torch.long) if pr is not None else None
        del W, L, R, pl, p, pr

    @property
    def weight(self):
        return form_full(self.L, self.R, self.pl, self.p, self.pr)

    def forward(self, x):
        x = form_full(self.L, self.R, self.pl, self.p, self.pr)[x]
        return x

    def assign(self, W, dtype, inp_x_out=False):
        if inp_x_out:
            assert (self.L.shape == W[0].shape and self.R.shape == W[1].shape)
            self.L = nn.Parameter(W[0].to(dtype))
            self.R = nn.Parameter(W[1].to(dtype))
        else:
            assert (self.L.shape == W[0].transpose(1, 2).shape and self.R.shape == W[1].transpose(1, 2).shape)
            self.L = nn.Parameter(W[0].transpose(1, 2).to(dtype))
            self.R = nn.Parameter(W[1].transpose(1, 2).to(dtype))


class MonarchOPTLearnedPositionalEmbedding(MonarchEmbedding):

    def __init__(self, kl, bl1, bl2, kr, br1, br2, use_pl=False, use_pr=False):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(kl, bl1, bl2, kr, br1, br2, use_pl=use_pl, use_pr=use_pr)

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
