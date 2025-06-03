import os
import torch
import torch.multiprocessing as mp
from slicegpt.utils import cleanup_memory
from .opp import procrustes_als
import time


def process_freq(frequency):
    return torch.sqrt(frequency + 1)


def init_attn_block(items):
    # initialize previous fc2 and attention input
    idx, layer_cfg, model_adapter, cfg, diag, device = items
    torch.set_num_threads(cfg.num_threads)
    layer_adapter = model_adapter.get_layers()[idx]
    prev_adapter = model_adapter.get_layers()[idx - 1] if idx > 0 else None
    qkv = layer_adapter.get_attention_inputs()
    joint_weight = [m.weight.data.T for m in qkv]
    params = [layer_cfg.get_params(idx, m, inp_x_out=True) for m in ['q_proj', 'k_proj', 'v_proj']]
    if prev_adapter is None:
        if isinstance(model_adapter.get_lm_head(), torch.nn.Linear):
            p = layer_cfg.get_params(-1, 'embed_tokens', inp_x_out=False)
            p['diag'] = process_freq(diag).to(device)
            joint_weight = [model_adapter.get_embeddings()[0].weight.data.T] + joint_weight
            params = [p] + params
    else:
        joint_weight = [prev_adapter.get_mlp_output().weight.data] + joint_weight
        if 'opt' in layer_cfg.cfg.model_name:
            params = [layer_cfg.get_params(idx - 1, 'fc2', inp_x_out=False)] + params
        elif 'llama' in layer_cfg.cfg.model_name:
            params = [layer_cfg.get_params(idx - 1, 'down_proj', inp_x_out=False)] + params
    Qt_1 = procrustes_als(joint_weight, params, cfg, name='attn_{}'.format(idx), layer_cfg=layer_cfg, device=device)
    del qkv, joint_weight, params, diag
    cleanup_memory()
    return Qt_1.cpu()


def assign_attn_block(idx, Qt_1, Qt_2, model_adapter, dtype, cfg, device):
    Qt_1, Qt_2 = Qt_1.to(device), Qt_2.to(device)
    layer_adapter = model_adapter.get_layers()[idx]
    prev_adapter = model_adapter.get_layers()[idx - 1] if idx > 0 else None
    if prev_adapter is None:
        # rotate embedding
        for emb in model_adapter.get_embeddings():
            emb.weight.data = (emb.weight.data.to(Qt_1.device, dtype=torch.float32) @ Qt_1.T).to(dtype).cpu()
    else:
        Q0 = prev_adapter.layer.mlp_shortcut_Q.data.to(Qt_1.device, dtype=torch.float32)
        prev_adapter.layer.mlp_shortcut_Q.data = (Qt_2 @ Q0 @ Qt_1.T).to(dtype).cpu()
        out = prev_adapter.get_mlp_output()
        out.weight.data = (Qt_1 @ out.weight.to(Qt_1.device, dtype=torch.float32)).to(dtype).cpu()
        if out.bias is not None:
            out.bias.data = (Qt_1 @ out.bias.to(Qt_1.device, dtype=torch.float32)).to(dtype).cpu()
        del Q0, out
    for i, module in enumerate(layer_adapter.get_attention_inputs()):
        module.weight.data = (module.weight.to(Qt_1.device, dtype=torch.float32) @ Qt_1.T).to(dtype).cpu()
    # clean memory
    del Qt_1, Qt_2
    cleanup_memory()


def init_ffn_block(items):
    idx, layer_cfg, model_adapter, cfg, diag, device = items
    torch.set_num_threads(cfg.num_threads)
    layer_adapter = model_adapter.get_layers()[idx]
    joint_weight = [layer_adapter.get_attention_output().weight.data] + \
                   [m.weight.data.T for m in layer_adapter.get_mlp_inputs()]
    params = [layer_cfg.get_params(idx, 'out_proj' if 'opt' in layer_cfg.cfg.model_name else 'o_proj', inp_x_out=False)]
    if 'opt' in layer_cfg.cfg.model_name:
        params += [layer_cfg.get_params(idx, 'fc1', inp_x_out=True)]
    elif 'llama' in layer_cfg.cfg.model_name:
        params += [layer_cfg.get_params(idx, 'gate_proj', inp_x_out=True),
                   layer_cfg.get_params(idx, 'up_proj', inp_x_out=True)]
    Qt_2 = procrustes_als(joint_weight, params, cfg, name='ffn_{}'.format(idx), layer_cfg=layer_cfg, device=device)
    del params, joint_weight, diag
    cleanup_memory()
    return Qt_2.cpu()


def assign_ffn_block(idx, Qt_1, Qt_2, model_adapter, dtype, cfg, device):
    Qt_1, Qt_2 = Qt_1.to(device), Qt_2.to(device)
    layer_adapter = model_adapter.get_layers()[idx]
    out = layer_adapter.get_attention_output()
    out.weight.data = (Qt_2 @ out.weight.to(Qt_1.device, dtype=torch.float32)).to(dtype).cpu()
    if out.bias is not None:
        out.bias.data = (Qt_2 @ out.bias.to(Qt_1.device, dtype=torch.float32)).to(dtype).cpu()
    Q0 = layer_adapter.layer.attn_shortcut_Q.data.to(Qt_1.device, dtype=torch.float32)
    layer_adapter.layer.attn_shortcut_Q.data = (Qt_1 @ Q0 @ Qt_2.T).to(dtype).cpu()
    for module in layer_adapter.get_mlp_inputs():
        module.weight.data = (module.weight.to(Qt_1.device, dtype=torch.float32) @ Qt_2.T).to(dtype).cpu()
    # clean memory
    del Qt_1, Qt_2, Q0, out
    cleanup_memory()
    # save('end init ffn block')


def init_head(items):
    # initialize last mlp output and lm_head
    idx, layer_cfg, model_adapter, cfg, diag, device = items
    torch.set_num_threads(cfg.num_threads)
    prev_adapter = model_adapter.get_layers()[-1]
    joint_weight = [prev_adapter.get_mlp_output().weight.data]
    if 'opt' in layer_cfg.cfg.model_name:
        params = [layer_cfg.get_params(idx, 'fc2', inp_x_out=False)]
    elif 'llama' in layer_cfg.cfg.model_name:
        params = [layer_cfg.get_params(idx, 'down_proj', inp_x_out=False)]
    if isinstance(model_adapter.get_lm_head(), torch.nn.Linear):
        joint_weight += [model_adapter.get_lm_head().weight.data.T]
        params += [layer_cfg.get_params(-1, 'lm_head', inp_x_out=True)]
        params[1]['diag'] = process_freq(diag).to(device)
    Qt_1 = procrustes_als(joint_weight, params, cfg, name='head', layer_cfg=layer_cfg, device=device)
    del joint_weight, params
    cleanup_memory()
    return Qt_1.cpu()


def assign_head(idx, Qt_1, Qt_2, model_adapter, dtype, cfg, device, Q_emb=None):
    Qt_1, Qt_2 = Qt_1.to(device), Qt_2.to(device)
    prev_adapter = model_adapter.get_layers()[-1]
    Q0 = prev_adapter.layer.mlp_shortcut_Q.data.to(Qt_1.device, dtype=torch.float32)
    prev_adapter.layer.mlp_shortcut_Q.data = (Qt_2 @ Q0 @ Qt_1.T).to(dtype).cpu()
    if isinstance(model_adapter.get_lm_head(), torch.nn.Linear):
        model_adapter.get_lm_head().weight.data = (
                model_adapter.get_lm_head().weight.to(Qt_1.device, dtype=torch.float32) @ Qt_1.T).to(dtype).cpu()
    else:
        Q_emb = Q_emb.to(device).to(torch.float32)
        d = model_adapter.get_lm_head().Q.to(device).to(torch.float32)
        if len(d.shape)==1:
            d = torch.diag(d)
        model_adapter.get_lm_head().Q = torch.nn.Parameter((Qt_1 @ d @ Q_emb.T).to(dtype).cpu())
        model_adapter.get_lm_head().b = torch.nn.Parameter((model_adapter.get_lm_head().b.to(device, torch.float32) @ Q_emb.T).to(dtype).cpu())
        model_adapter.get_lm_head().e = torch.nn.Parameter((Q_emb @ model_adapter.get_lm_head().e.to(device, torch.float32)).to(dtype).cpu())
        del d, Q_emb
    out = prev_adapter.get_mlp_output()
    out.weight.data = (Qt_1 @ out.weight.data.to(Qt_1.device, dtype=torch.float32)).to(dtype).cpu()
    if out.bias is not None:
        out.bias.data = (Qt_1 @ out.bias.data.to(Qt_1.device, dtype=torch.float32)).to(dtype).cpu()
    # clean memory
    del Qt_2, Qt_1, prev_adapter, out, Q0
    # print('end init head')
    cleanup_memory()


def procrustes_init(model_adapter, cfg, layer_cfg, diag, device, log=False):
    dtype = next(iter(model_adapter.model.parameters())).dtype
    num_layers = len(model_adapter.get_layers())
    # model_adapter.model.share_memory()
    lst = [(idx, layer_cfg, model_adapter, cfg, diag, device) for idx in range(num_layers)]
    if log:
        print('mem before mp.Pool', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
    start_time = time.time()
    Qt1_0 = init_attn_block(lst[0])  # requires more memory, run this separately
    with mp.Pool(processes=cfg.n_process) as pool:
        # project previous output and attention inputs
        Qt1_lst = pool.map(init_attn_block, lst[1:])
        # project attention output and mpl input
        Qt2_lst = pool.map(init_ffn_block, lst)
        # initialize last fc2 and lm_head
    Q_head = init_head(
        (num_layers - 1, layer_cfg, model_adapter, cfg, diag, device))  # requires more memory, run this separately
    del lst, diag
    cleanup_memory()
    if log:
        print('mem after mp.Pool', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
        print('Time', time.time() - start_time, flush=True)

    # save to disk
    if not os.path.exists(layer_cfg.cfg.tmp_path):
        os.mkdir(layer_cfg.cfg.tmp_path)
    torch.save(Qt1_0, os.path.join(layer_cfg.cfg.tmp_path, 'Qt1_idx_0.pt'))
    for i in range(len(Qt1_lst)):
        torch.save(Qt1_lst[i], os.path.join(layer_cfg.cfg.tmp_path, 'Qt1_idx_{}.pt'.format(i + 1)))
    for i in range(len(Qt2_lst)):
        torch.save(Qt2_lst[i], os.path.join(layer_cfg.cfg.tmp_path, 'Qt2_idx_{}.pt'.format(i)))
    torch.save(Q_head, os.path.join(layer_cfg.cfg.tmp_path, 'Q_head.pt'))

    # W_in -> W_in Q, W_out -> Q.T W_out
    assign_attn_block(0, Qt1_0, Qt2_lst[-1], model_adapter, dtype, cfg, device)
    assign_ffn_block(0, Qt1_0, Qt2_lst[0], model_adapter, dtype, cfg, device)
    for idx in range(1, num_layers):
        assign_attn_block(idx, Qt1_lst[idx - 1], Qt2_lst[idx - 1], model_adapter, dtype, cfg, device)
        assign_ffn_block(idx, Qt1_lst[idx - 1], Qt2_lst[idx], model_adapter, dtype, cfg, device)
    assign_head(-1, Q_head, Qt2_lst[-1], model_adapter, dtype, cfg, device, Q_emb=Qt1_0)
    if log:
        print('mem after assignment', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
    del Qt1_0, Q_head, Qt2_lst, Qt1_lst, cfg, layer_cfg
    cleanup_memory()
    return model_adapter
