import torch
import torch.nn as nn
from slicegpt.utils import cleanup_memory, map_tensors
from .wopp import weighted_procrustes_als
from .kron_utils import kron_project
from .monarch_utils import monarch_project
from .layers_replacement import replace_linear, replace_embedding
import time


def process_freq(frequency):
    return torch.sqrt(frequency + 1)


def init_attention_block(idx, model_name, layer_adapter, prev_adapter, model_adapter,
                         dtype, layer_cfg, cfg, Qt_2, Xout, Xin, device, cut_values,
                         frequency, compress_emb=True):
    print('start init_attention_block', flush=True)
    # project previous output and attention inputs
    Win = [m.weight.data.T for m in layer_adapter.get_attention_inputs()]
    params_in = [layer_cfg.get_params(idx, m, inp_x_out=True) for m in ['q_proj', 'k_proj', 'v_proj']]
    Wout, params_out, diag, Q0 = None, None, None, None
    if prev_adapter is None:
        diag = process_freq(frequency)
        for i, m in enumerate(model_adapter.get_embeddings()):
            p = layer_cfg.get_params(-1, 'embed_tokens', inp_x_out=True)
            if i == 0:
                diag = diag.to(device)
                p['diag'] = diag
                Wout = [m.weight.data]
                params_out = [p]
    else:
        Q0 = prev_adapter.layer.mlp_shortcut_Q.to(Qt_2.device, dtype=torch.float32)
        Wout = [prev_adapter.get_mlp_output().weight.data.T]
        name = 'fc2' if 'opt' in layer_cfg.cfg.model_name else 'down_proj'
        params_out = [layer_cfg.get_params(idx, name, inp_x_out=True)]
    als_iters = cfg.als_iters
    if prev_adapter is None:
        cfg.als_iters = 0
    Qt_1, Wout_appr, Win_appr, Q_skip = weighted_procrustes_als(Wout, Xout, Win, Xin, params_out, params_in,
                                                                cfg, layer_cfg, device, cut_values=cut_values,
                                                                diag_emb=diag)
    cfg.als_iters = als_iters
    if prev_adapter is None:
        del Wout_appr
        Wout_appr = []
        names = ['embed_tokens', 'embed_positions'] if 'opt' in layer_cfg.cfg.model_name else ['embed_tokens']
        if compress_emb:
            for name, emb in zip(names, model_adapter.get_embeddings()):
                if layer_cfg.cfg.decomposition == 'kronecker':
                    Wout_appr += kron_project([emb.weight.data.to(Qt_1.device, torch.float32) @ Qt_1.T],
                                              [layer_cfg.get_params(-1, name, inp_x_out=True)])
                elif layer_cfg.cfg.decomposition == 'monarch':
                    Wout_appr += monarch_project([emb.weight.data.to(Qt_1.device, torch.float32) @ Qt_1.T],
                                                 [layer_cfg.get_params(-1, name, inp_x_out=True)])

                else:
                    raise ValueError()
            replace_embedding(model_adapter.model, layer_cfg.cfg, dtype, init=False)
            for i, emb in enumerate(model_adapter.get_embeddings()):
                emb.assign(Wout_appr[i], dtype=dtype, inp_x_out=True)
        else:
            for emb in model_adapter.get_embeddings():
                emb.weight.data = (emb.weight.data.to(Qt_1.device, torch.float32) @ Qt_1.T).to(dtype)
    else:
        bias = None
        if prev_adapter.get_mlp_output().bias is not None:
            bias = prev_adapter.get_mlp_output().bias.data
        name = 'fc2' if 'opt' in layer_cfg.cfg.model_name else 'down_proj'
        replace_linear(name, idx - 1, model_adapter.model, layer_cfg.cfg, dtype, init=False)
        if Q_skip is not None:
            del prev_adapter.layer.mlp_shortcut_Q
            prev_adapter.layer.mlp_shortcut_Q = Q_skip
        else:
            prev_adapter.layer.mlp_shortcut_Q = nn.Parameter((Qt_2 @ Q0 @ Qt_1.T).to(dtype=dtype))
        prev_adapter.get_mlp_output().assign(Wout_appr[0], bias, dtype=dtype, inp_x_out=True, Qt=Qt_1)
        del Q0, bias
    bias = [m.bias for m in layer_adapter.get_attention_inputs()]
    for name in ['q_proj', 'k_proj', 'v_proj']:
        replace_linear(name, idx, model_adapter.model, layer_cfg.cfg, dtype, init=False)
    for i, module in enumerate(layer_adapter.get_attention_inputs()):
        module.assign(Win_appr[i], bias[i], dtype=dtype, inp_x_out=True)
    # clean memory
    del Wout, Win, Xout, Xin, params_out, params_in, Wout_appr, Win_appr, Qt_2, prev_adapter, model_adapter, \
        frequency, bias, Q_skip
    cleanup_memory()
    print('end init_attention_block', flush=True)
    return Qt_1


def init_ffn_block(idx, layer_adapter, model_adapter, dtype, layer_cfg, cfg, Qt_1, Xout, Xin, device):
    print('start init_ffn_block', flush=True)
    # project attention output and mpl input
    Wout = [layer_adapter.get_attention_output().weight.data.T]
    Win = [m.weight.data.T for m in layer_adapter.get_mlp_inputs()]
    name_out = 'out_proj' if 'opt' in layer_cfg.cfg.model_name else 'o_proj'
    params_out = [layer_cfg.get_params(idx, name_out, inp_x_out=True)]
    names_in = ['fc1'] if 'opt' in layer_cfg.cfg.model_name else ['gate_proj', 'up_proj']
    params_in = []
    for name in names_in:
        params_in += [layer_cfg.get_params(idx, name, inp_x_out=True)]
    Q0 = layer_adapter.layer.attn_shortcut_Q.data.to(Qt_1.device, dtype=torch.float32)
    Qt_2, Wout_appr, Win_appr, Q_skip = weighted_procrustes_als(Wout, Xout, Win, Xin, params_out, params_in,
                                                                cfg, layer_cfg, device, cut_values=False)
    if Q_skip is not None:
        del layer_adapter.layer.attn_shortcut_Q
        layer_adapter.layer.attn_shortcut_Q = Q_skip
    else:
        layer_adapter.layer.attn_shortcut_Q = nn.Parameter((Qt_1 @ Q0 @ Qt_2.T).to(dtype=dtype))
    bias = layer_adapter.get_attention_output().bias
    replace_linear(name_out, idx, model_adapter.model, layer_cfg.cfg, dtype, init=False)
    layer_adapter.get_attention_output().assign(Wout_appr[0], bias, dtype=dtype, inp_x_out=True, Qt=Qt_2)
    bias = [m.bias for m in layer_adapter.get_mlp_inputs()]
    for name in names_in:
        replace_linear(name, idx, model_adapter.model, layer_cfg.cfg, dtype, init=False)
    for i, module in enumerate(layer_adapter.get_mlp_inputs()):
        module.assign(Win_appr[i], bias[i], dtype=dtype, inp_x_out=True)
    # clean memory
    del Wout, Win, params_out, params_in, Wout_appr, Win_appr, Qt_1, Xout, Xin, Q0, bias, Q_skip
    cleanup_memory()
    print('end init_ffn_block', flush=True)
    return Qt_2


def init_head(idx, prev_adapter, model_adapter, dtype, layer_cfg, cfg, Qt_2, Xout, Xin, device,
              frequency, Q_emb=None):
    print('start init_head', flush=True)
    # initialize last fc and lm head
    Win, params_in, head_weight = None, None, None
    if isinstance(model_adapter.get_lm_head(), nn.Linear):
        head_weight = model_adapter.get_lm_head().weight.data.T
        Win = [head_weight]
        params_in = [layer_cfg.get_params(-1, 'lm_head', inp_x_out=True)]
    Wout = [prev_adapter.get_mlp_output().weight.data.T]
    name = 'fc2' if 'opt' in layer_cfg.cfg.model_name else 'down_proj'
    params_out = [layer_cfg.get_params(idx-1, name, inp_x_out=True)]
    Q0 = prev_adapter.layer.mlp_shortcut_Q.data.to(Qt_2.device, dtype=torch.float32)
    Qt_1, Wout_appr, Win_appr, Q_skip = weighted_procrustes_als(Wout, Xout, Win, Xin, params_out, params_in,
                                                                cfg, layer_cfg, device, cut_values=False)
    bias = prev_adapter.get_mlp_output().bias
    replace_linear(name, idx - 1, model_adapter.model, layer_cfg.cfg, dtype, init=False)
    prev_adapter.get_mlp_output().assign(Wout_appr[0], bias, dtype=dtype, inp_x_out=True, Qt=Qt_1)
    if isinstance(model_adapter.get_lm_head(), nn.Linear):
        replace_linear('lm_head', -1, model_adapter.model, layer_cfg.cfg, dtype, init=False)
        model_adapter.get_lm_head().assign(Win_appr[0], model_adapter.get_lm_head().bias, dtype=dtype, inp_x_out=True)
    else:
        Q_emb = Q_emb.to(device).to(torch.float32)
        d = model_adapter.get_lm_head().Q.to(device).to(torch.float32)
        if len(d.shape)==1:
            d = torch.diag(d)
        model_adapter.get_lm_head().Q = torch.nn.Parameter((Qt_1 @ d @ Q_emb.T).to(dtype).cpu())
        model_adapter.get_lm_head().b = torch.nn.Parameter((model_adapter.get_lm_head().b.to(device, torch.float32) @ Q_emb.T).to(dtype).cpu())
        model_adapter.get_lm_head().e = torch.nn.Parameter((Q_emb @ model_adapter.get_lm_head().e.to(device, torch.float32)).to(dtype).cpu())
        del d
    if Q_skip is not None:
        del prev_adapter.layer.mlp_shortcut_Q
        prev_adapter.layer.mlp_shortcut_Q = Q_skip
    else:
        prev_adapter.layer.mlp_shortcut_Q = nn.Parameter((Qt_2 @ Q0 @ Qt_1.T).to(dtype=dtype))
    # clean memory
    del Qt_2, Qt_1, Q0, head_weight, Wout, Win, params_out, params_in, Wout_appr, Win_appr, prev_adapter, \
        Xin, Xout, bias, Q_skip, frequency, Q_emb
    cleanup_memory()
    print('end init_head', flush=True)


def weighted_procrustes_init(model_name, model_adapter, dataloader, layer_cfg, cfg, frequency, device, cut_values,
                             compress_emb=True, log=False):
    # |Q Wout Xout - Wout' Xout'|_F + |XTin Q WTin - XTin' WTin'|_F -> min;
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    prev_adapter, Qt_2, prev_mlp_out, Xskip, Q_emb = None, None, None, None, None
    layers = model_adapter.get_layers()
    if log:
        print('mem start', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
    inps, args, kwargs, ignore_masks = get_args(dataloader, cfg, model_adapter, device)
    # get inputs to layer 0
    for i, inp in enumerate(inps):
        args[i] = layers[0].get_updated_args(inp, args[i])
    attn_inp, attn_out, mlp_inp, mlp_out, inps = get_inputs(layers[0], args, kwargs, ignore_masks,
                                                            device, cfg.num_samples,
                                                            decomposition=layer_cfg.cfg.decomposition)
    for idx, layer_adapter in enumerate(layers):
        print('layer {}'.format(idx), torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)

        start_time = time.time()
        if idx + 1 < len(layers):
            # update arguments to next layer
            for i, inp in enumerate(inps):
                args[i] = layers[idx + 1].get_updated_args(inp, args[i])
            del inps
            # get inputs to next layer
            next_inputs = get_inputs(layers[idx + 1], args, kwargs, ignore_masks, device, cfg.num_samples,
                                     decomposition=layer_cfg.cfg.decomposition)
        # project previous output and attention inputs
        Qt_1 = init_attention_block(idx, model_name, layer_adapter, prev_adapter, model_adapter,
                                    dtype, layer_cfg, cfg, Qt_2, prev_mlp_out, attn_inp,
                                    device, cut_values, frequency, compress_emb)
        Q_emb = Qt_1 if Q_emb is None else Q_emb
        # project attention output and mpl input
        del attn_inp, prev_mlp_out
        Qt_2 = init_ffn_block(idx, layer_adapter, model_adapter, dtype, layer_cfg, cfg, Qt_1, attn_out, mlp_inp, device)
        prev_mlp_out = mlp_out
        # clean memory
        del Qt_1, attn_out, mlp_inp, mlp_out
        cleanup_memory()
        if idx + 1 < len(layers):
            attn_inp, attn_out, mlp_inp, mlp_out, inps = next_inputs
            del next_inputs
        prev_adapter = layer_adapter
        if log:
            print('Time', time.time() - start_time, flush=True)
    # initialize last fc2 and lm_head
    head_inp = reduce_output(inps, device, decomposition=layer_cfg.cfg.decomposition)
    del inps
    init_head(len(layers), prev_adapter, model_adapter, dtype, layer_cfg, cfg, Qt_2, prev_mlp_out, head_inp, device,
              frequency, Q_emb=Q_emb)
    del head_inp, prev_mlp_out, Qt_2, frequency, Q_emb
    cleanup_memory()
    return model_adapter


@torch.no_grad()
def get_inputs(layer_adapter, layer_args: list, layer_kwargs: list, ignore_masks=None,
               device=torch.device('cpu'), max_samples=1, decomposition='kronecker', log=False):
    if log:
        print('mem start get_inputs', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
    layer_adapter.layer.to(device)
    layer_adapter.layer.eval()
    attn_inp = [layer_adapter.get_attention_inputs()[0]]
    attn_out = [layer_adapter.get_attention_output()]
    mlp_inp = [layer_adapter.get_mlp_inputs()[0]]
    mlp_out = [layer_adapter.get_mlp_output()]

    attn_inp_list = [[None]]
    mlp_inp_list = [[None]]
    attn_out_list = [[None]]
    mlp_out_list = [[None]]
    outputs = []
    max_positions = 2048

    def hook_inp(tens):

        def hook_inp_fn(_, args: tuple, _output) -> None:
            nonlocal tens, max_positions
            inp = args[0].detach()
            inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
            if tens[0] is None:
                tens[0] = (inp.T / max_samples) @ (inp / max_positions)
            else:
                tens[0] += (inp.T / max_samples) @ (inp / max_positions)
            del inp, args, _output

        return hook_inp_fn

    hooks = []
    for i in range(len(attn_inp)):
        hook = attn_inp[i].register_forward_hook(hook_inp(attn_inp_list[i]))
        hooks.append(hook)
    # if len(attn_inp) == 3:
    #    hook = attn_inp[-1].register_forward_hook(hook_attn(attn_inp_list[-1]))
    #    hooks.append(hook)
    for i in range(len(mlp_inp)):
        hook = mlp_inp[i].register_forward_hook(hook_inp(mlp_inp_list[i]))
        hooks.append(hook)
    hook = attn_out[0].register_forward_hook(hook_inp(attn_out_list[0]))
    hooks.append(hook)
    hook = mlp_out[0].register_forward_hook(hook_inp(mlp_out_list[0]))
    hooks.append(hook)

    n_samples = 0
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=device
        )
        batch_size = layer_args_batch[0].shape[0]
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        outputs.append(out.detach().cpu())
        del layer_args_batch, layer_kwargs_batch
        cleanup_memory()

        n_samples += out.shape[0]
        if n_samples >= max_samples:
            break

    for hook in hooks:
        hook.remove()
    del hook, hooks
    attn_inp_res, attn_out_res, mlp_inp_res, mlp_out_res = [], [], [], []
    attn_inp_inv, attn_out_inv, mlp_inp_inv, mlp_out_inv = [], [], [], []
    eps = 1e-5
    for M in attn_inp_list:
        U, S, Vt = torch.linalg.svd(M[0].to(device), full_matrices=False)
        attn_inp_res.append((Vt.T * torch.sqrt(S)[None, :]) @ Vt)
        S_inv = torch.zeros_like(S)
        mask = (S >= eps)
        if torch.sum(mask) < len(mask) or decomposition != 'kronecker':
            attn_inp_inv.append(None)
        else:
            S_inv[mask] = 1 / S[mask]
            attn_inp_inv.append((Vt.T * S_inv[None, :]) @ Vt)
        del U, S, Vt, S_inv, mask
        cleanup_memory()
    attn_inp_res.append(attn_inp_res[-1].clone())
    attn_inp_res.append(attn_inp_res[-1].clone())
    attn_inp_inv.append(attn_inp_inv[-1].clone() if attn_inp_inv[-1] is not None else None)
    attn_inp_inv.append(attn_inp_inv[-1].clone() if attn_inp_inv[-1] is not None else None)
    for M in mlp_inp_list:
        U, S, Vt = torch.linalg.svd(M[0].to(device), full_matrices=False)
        mlp_inp_res.append((Vt.T * torch.sqrt(S)[None, :]) @ Vt)
        S_inv = torch.zeros_like(S)
        mask = (S >= eps)
        if torch.sum(mask) < len(mask) or decomposition != 'kronecker':
            mlp_inp_inv.append(None)
        else:
            S_inv[mask] = 1 / S[mask]
            mlp_inp_inv.append((Vt.T * S_inv[None, :]) @ Vt)
        del U, S, Vt, S_inv, mask
        cleanup_memory()
    if len(layer_adapter.get_mlp_inputs()) > 1:
        mlp_inp_res.append(mlp_inp_res[-1].clone())
        mlp_inp_inv.append(mlp_inp_inv[-1].clone() if mlp_inp_inv[-1] is not None else None)
    U, S, Vt = torch.linalg.svd(attn_out_list[0][0].to(device), full_matrices=False)
    attn_out_res = [(Vt.T * torch.sqrt(S)[None, :]) @ Vt]
    S_inv = torch.zeros_like(S)
    mask = (S >= eps)
    if torch.sum(mask) < len(mask) or decomposition != 'kronecker':
        attn_out_inv = [None]
    else:
        S_inv[mask] = 1 / S[mask]
        attn_out_inv = [(Vt.T * S_inv[None, :]) @ Vt]
    del U, S, Vt, S_inv
    cleanup_memory()
    U, S, Vt = torch.linalg.svd(mlp_out_list[0][0].to(device), full_matrices=False)
    mlp_out_res = [(Vt.T * torch.sqrt(S)[None, :]) @ Vt]
    S_inv = torch.zeros_like(S)
    mask = (S >= eps)
    if torch.sum(mask) < len(mask) or decomposition != 'kronecker':
        mlp_out_inv = [None]
    else:
        S_inv[mask] = 1 / S[mask]
        mlp_out_inv = [(Vt.T * S_inv[None, :]) @ Vt]
    layer_adapter.layer.cpu()
    del S_inv, mask, U, S, Vt, attn_inp, attn_out, mlp_inp, mlp_out, attn_inp_list, attn_out_list, mlp_inp_list, mlp_out_list, layer_adapter
    cleanup_memory()
    return (attn_inp_res, attn_inp_inv), (attn_out_res, attn_out_inv), \
           (mlp_inp_res, mlp_inp_inv), (mlp_out_res, mlp_out_inv), outputs


def reduce_output(outputs, device, max_positions=2048, decomposition='kronecker'):
    if outputs is None:
        return [None]
    M = 0
    for out in outputs:
        out = out.detach().reshape(-1, out.shape[-1]).to(device, dtype=torch.float32)
        M += (out.T / len(outputs)) @ (out / max_positions)
        del out
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    result = [(Vt.T * torch.sqrt(S)[None, :]) @ Vt]
    inverted = [None]
    S_inv, mask = None, None
    if decomposition == 'kronecker':
        eps = 1e-6
        S_inv = torch.zeros_like(S)
        mask = (S >= eps)
        S_inv[mask] = 1 / S[mask]
        inverted = [(Vt.T * S_inv[None, :]) @ Vt]
    del U, S, Vt, M, outputs, S_inv, mask
    return (result, inverted)


@torch.no_grad()
def get_args(dataloader, cfg, model_adapter, device, apply_mask=False, log=False):
    # Get the input of the first layer norm
    if log:
        print('mem start get_args', torch.cuda.memory_allocated() / 1024 / 1024 / 1024,  flush=True)
    inps, args, kwargs, ignore_masks = [], [], [], []
    cnt = 0
    model_adapter.model.to(device)
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_args(model_adapter, batch, device)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])
        cnt += batch['input_ids'].shape[0]
        del inp_batch, batch
        cleanup_memory()
        if cnt > cfg.num_samples:
            break
    model_adapter.model.cpu()
    if log:
        print('end get_args', torch.cuda.memory_allocated() / 1024 / 1024 / 1024,  flush=True)
    return inps, args, kwargs, ignore_masks


@torch.no_grad()
def get_layer0_args(model_adapter, batch, device):  # -> tuple[Tensor, tuple, dict[str, Any]]:

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    # for emb in model_adapter.get_embeddings():
    #    emb.to(device)
    # layer0_adapter.layer.to(device)
    layer0_adapter.layer.eval()
    layer0_catcher = Catcher()
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        batch = map_tensors(batch, device=device)
        model_adapter.model(**batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = map_tensors(args, device='cpu')
    kwargs = map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # return to cpu
    # for emb in model_adapter.get_embeddings():
    #    emb.cpu()
    # layer0_adapter.layer.cpu()

    # Run GC and cleanup GPU memory
    cleanup_memory()

    return args[layer0_adapter.hidden_states_args_position], args, kwargs
