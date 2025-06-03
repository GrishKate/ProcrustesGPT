import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import Conv1D
from transformers.models.opt.modeling_opt import OPTAttention, OPTLearnedPositionalEmbedding
from transformers.models.llama.modeling_llama import LlamaAttention
from structured_layers import *

lin_types = [nn.Linear, Conv1D]
attn_types = [OPTAttention, LlamaAttention]
embed_types = [nn.Embedding, OPTLearnedPositionalEmbedding]


def replace_all_layers(model, cfg_path, model_name, init=False, compress_head_emb=True, dtype=torch.float16):
    cfg = OmegaConf.load(cfg_path)
    special_layers = [k for k in cfg.not_replace_layers.keys()] if cfg.not_replace_layers is not None else []
    for name, l in model.named_modules():
        if name == '' or name in special_layers:
            continue
        m = model
        names = name.split(".")
        for i in range(len(names) - 1):
            m = getattr(m, names[i])
        for lin_type in lin_types:
            if isinstance(l, lin_type):
                if names[-1] == 'lm_head' and not compress_head_emb:
                    continue
                w = l.weight.shape
                if 'kronecker' in cfg.decomposition:
                    r, m1, n1 = cfg[names[-1]]["r"], cfg[names[-1]]["m1"], cfg[names[-1]]["n1"]
                    if names[-1] != 'lm_head':
                        idx = int(names[3]) if 'opt' in model_name else int(names[2])
                        if idx not in cfg.layer_numbers:
                            r, m1, n1 = 1, 1, 1
                            init = True
                    m2, n2 = w[1] // m1, w[0] // n1
                    params = {'r': r, 'm1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'decomposition': cfg.decomposition}
                    layer_cls = KronLinear
                elif cfg.decomposition == 'monarch':
                    kl, bl2, kr, br1 = cfg[names[-1]]["kl"], cfg[names[-1]]["bl2"], cfg[names[-1]]["kr"], \
                                       cfg[names[-1]]["br1"]
                    bl1, br2 = cfg[names[-1]]["bl1"], cfg[names[-1]]["br2"]
                    use_pl, use_pr = cfg[names[-1]]["use_pl"], cfg[names[-1]]["use_pr"]
                    params = {'kl': kl, 'bl1': bl1, 'bl2': bl2, 'kr': kr, 'br1': br1, 'br2': br2,
                              'use_pl': use_pl, 'use_pr': use_pr}
                    layer_cls = MonarchLinear
                else:
                    raise ValueError('unknown decomposition')
                setattr(
                    m,
                    names[-1],
                    layer_cls.get_from_layer(
                        l, params,
                        init=init,
                        dtype=dtype
                    ),
                )
                break
        for embed_type in embed_types:
            if isinstance(l, embed_type):
                if names[-1] in special_layers or not compress_head_emb:
                    continue
                w = l.weight.shape
                if 'kronecker' in cfg.decomposition:
                    r, m1, n1 = cfg[names[-1]]["r"], cfg[names[-1]]["m1"], cfg[names[-1]]["n1"]
                    m2, n2 = w[0] // m1, w[1] // n1
                    params = {'r': r, 'm1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'decomposition': cfg.decomposition}
                    layer_cls = KronEmbedding
                    if isinstance(l, OPTLearnedPositionalEmbedding):
                        layer_cls = KronOPTLearnedPositionalEmbedding
                elif cfg.decomposition == 'monarch':
                    kl, bl2, kr, br1 = cfg[names[-1]]["kl"], cfg[names[-1]]["bl2"], cfg[names[-1]]["kr"], \
                                       cfg[names[-1]]["br1"]
                    bl1, br2 = cfg[names[-1]]["bl1"], cfg[names[-1]]["br2"]
                    use_pl, use_pr = cfg[names[-1]]["use_pl"], cfg[names[-1]]["use_pr"]
                    params = {'kl': kl, 'bl1': bl1, 'bl2': bl2, 'kr': kr, 'br1': br1, 'br2': br2,
                              'use_pl': use_pl, 'use_pr': use_pr}
                    layer_cls = MonarchEmbedding
                    if isinstance(l, OPTLearnedPositionalEmbedding):
                        layer_cls = MonarchOPTLearnedPositionalEmbedding
                else:
                    raise ValueError('unknown decomposition')
                print(name, 'shape is', w)
                setattr(
                    m,
                    names[-1],
                    layer_cls.get_from_layer(
                        l, params,
                        init=init,
                        dtype=dtype,
                    ),
                )
                break
    return None


def replace_embedding(model, cfg, dtype, init=False):
    special_layers = [k for k in cfg.not_replace_layers.keys()] if cfg.not_replace_layers is not None else []
    for name, l in model.named_modules():
        if name == '' or name in special_layers:
            continue
        m = model
        names = name.split(".")
        for i in range(len(names) - 1):
            m = getattr(m, names[i])
        for embed_type in embed_types:
            if isinstance(l, embed_type):
                if names[-1] in special_layers:
                    continue
                w = l.weight.shape
                if cfg.decomposition == 'kronecker':
                    r, m1, n1 = cfg[names[-1]]["r"], cfg[names[-1]]["m1"], cfg[names[-1]]["n1"]
                    m2, n2 = w[0] // m1, w[1] // n1
                    params = {'r': r, 'm1': m1, 'n1': n1, 'm2': m2, 'n2': n2}
                    layer_cls = KronEmbedding
                    if isinstance(l, OPTLearnedPositionalEmbedding):
                        layer_cls = KronOPTLearnedPositionalEmbedding
                elif cfg.decomposition == 'monarch':
                    kl, bl2, kr, br1 = cfg[names[-1]]["kl"], cfg[names[-1]]["bl2"], cfg[names[-1]]["kr"], \
                                       cfg[names[-1]]["br1"]
                    bl1, br2 = cfg[names[-1]]["bl1"], cfg[names[-1]]["br2"]
                    use_pl, use_pr = cfg[names[-1]]["use_pl"], cfg[names[-1]]["use_pr"]
                    params = {'kl': kl, 'bl1': bl1, 'bl2': bl2, 'kr': kr, 'br1': br1, 'br2': br2,
                              'use_pl': use_pl, 'use_pr': use_pr}
                    layer_cls = MonarchEmbedding
                    if isinstance(l, OPTLearnedPositionalEmbedding):
                        layer_cls = MonarchOPTLearnedPositionalEmbedding
                else:
                    raise ValueError('unknown decomposition')
                print(name, 'shape is', w)
                setattr(
                    m,
                    names[-1],
                    layer_cls.get_from_layer(
                        l, params,
                        init=init,
                        dtype=dtype,
                    ),
                )
                break


def replace_linear(layer_name, idx, model, cfg, dtype, init=False):
    special_layers = [k for k in cfg.not_replace_layers.keys()] if cfg.not_replace_layers is not None else []
    for name, l in model.named_modules():
        if name == '' or name in special_layers:
            continue
        m = model
        names = name.split(".")
        for i in range(len(names) - 1):
            m = getattr(m, names[i])
        for lin_type in lin_types:
            if isinstance(l, lin_type) and names[-1] == layer_name:
                w = l.weight.shape
                if names[-1] != 'lm_head':
                    cnt = int(names[3]) if 'opt' in cfg.model_name else int(names[2])
                    if cnt != idx or cnt not in cfg.layer_numbers:
                        continue
                if cfg.decomposition == 'kronecker':
                    r, m1, n1 = cfg[names[-1]]["r"], cfg[names[-1]]["m1"], cfg[names[-1]]["n1"]
                    m2, n2 = w[1] // m1, w[0] // n1
                    params = {'r': r, 'm1': m1, 'n1': n1, 'm2': m2, 'n2': n2}
                    layer_cls = KronLinear
                elif cfg.decomposition == 'monarch':
                    kl, bl2, kr, br1 = cfg[names[-1]]["kl"], cfg[names[-1]]["bl2"], cfg[names[-1]]["kr"], \
                                       cfg[names[-1]]["br1"]
                    bl1, br2 = cfg[names[-1]]["bl1"], cfg[names[-1]]["br2"]
                    use_pl, use_pr = cfg[names[-1]]["use_pl"], cfg[names[-1]]["use_pr"]
                    params = {'kl': kl, 'bl1': bl1, 'bl2': bl2, 'kr': kr, 'br1': br1, 'br2': br2,
                              'use_pl': use_pl, 'use_pr': use_pr}
                    layer_cls = MonarchLinear
                else:
                    raise ValueError('unknown decomposition')
                setattr(
                    m,
                    names[-1],
                    layer_cls.get_from_layer(
                        l, params,
                        init=init,
                        dtype=dtype
                    ),
                )
                break
