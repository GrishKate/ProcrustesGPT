# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion
from slicegpt.utils import cleanup_memory

from .layers_replacement import replace_all_layers
from .init_frobenius import procrustes_init, assign_attn_block, assign_ffn_block, assign_head
from .init_weighted import weighted_procrustes_init


def procrustes_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--calibration_dataset", type=str, help="Calibration dataset", choices=["wikitext2"],
                        default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--init_from_saved', type=bool, default=True)
    parser.add_argument('--device', type=str, default=None,
                        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.", )
    parser.add_argument('--cfg_for_layers_path', type=str, default='./configs/cfg_layers.yaml')
    parser.add_argument('--cfg_for_compression_path', type=str, default='./configs/cfg_compression.yaml')

    parser.add_argument('--save_rotated', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--filename', type=str, default='model.pt')

    parser.add_argument('--skip_connections', type=str, default=None, choices=['exponent', 'cayley'])

    parser.add_argument("--ppl_eval_dataset", type=str, help="Calibration dataset", choices=["wikitext2"],
                        default="wikitext2")
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--distribute-model", action="store_true",
                        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.", )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_args(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device:
        device = torch.device(args.device)

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")
    return dtype, device


class LayersConfig:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_params(self, idx, name, inp_x_out=True):
        if name == 'skip':
            p = {n: self.cfg[name][n] for n in ['kl', 'kr', 'bl1', 'bl2', 'br1', 'br2', 'use_pl', 'use_pr']}
            p['inp_x_out'] = True
            return p
        if 'kronecker' in self.cfg.decomposition:
            r, m1, n1 = self.cfg[name]['r'], self.cfg[name]['m1'], self.cfg[name]['n1']
            if not inp_x_out:
                m1, n1 = n1, m1
            if name not in ['lm_head', 'embed_tokens', 'embed_positions'] and idx not in self.cfg.layer_numbers:
                r, m1, n1 = 1, 1, 1
            return {'r': r, 'm1': m1, 'n1': n1, 'inp_x_out': inp_x_out}
        elif self.cfg.decomposition == 'monarch':
            kl, bl1, bl2 = self.cfg[name]['kl'], self.cfg[name]['bl1'], self.cfg[name]['bl2']
            kr, br1, br2 = self.cfg[name]['kr'], self.cfg[name]['br1'], self.cfg[name]['br2']
            use_pl, use_pr = self.cfg[name]['use_pl'], self.cfg[name]['use_pr']
            if not inp_x_out:
                kl, bl1, bl2, kr, br1, br2 = kr, br2, br1, kl, bl2, bl1
                use_pl, use_pr = use_pr, use_pl
            # if name not in ['lm_head', 'embed_tokens', 'embed_positions'] and idx not in self.cfg.layer_numbers:
            #    r, m1, n1 = 1, 1, 1
            return {'kl': kl, 'bl1': bl1, 'bl2': bl2, 'kr': kr, 'br1': br1,
                    'br2': br2, 'use_pl': use_pl, 'use_pr': use_pr, 'inp_x_out': inp_x_out}
        else:
            raise NotImplementedError('unknown decomposition')


def get_rotated_adapter(args, log=False):
    dtype, device = process_args(args)
    with open(args.cfg_for_compression_path, 'r') as f:
        print(f.read(), flush=True)
    cfg = OmegaConf.load(args.cfg_for_compression_path)
    layer_cfg = LayersConfig(OmegaConf.load(args.cfg_for_layers_path))
    # load the pre-trained model
    if log:
        print('mem start get_rotated_adapter', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model_name, args.model_path, token=args.hf_token,
                                                                dtype=dtype)
    original_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
    print('Loaded adapter', flush=True)
    # replace modules with compressible equivalents
    layernorm_fusion.replace_layers(model_adapter)
    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(model_adapter, detach_head=layer_cfg.cfg.detach_head)
    print('Fused modules', flush=True)
    after_fusion_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
    # add matrices to skip connections
    hidden_size = model_adapter.hidden_size
    for layer_adapter in model_adapter.get_layers():
        layer_adapter.layer.attn_shortcut_Q = nn.Parameter(torch.eye(hidden_size, device=device))
        layer_adapter.layer.mlp_shortcut_Q = nn.Parameter(torch.eye(hidden_size, device=device))
    print('Model name', args.model_name, flush=True)
    print(f'Original model parameters: {original_param_count:,d}', flush=True)
    print(f'After fusion parameters: {after_fusion_param_count:,d}', flush=True)
    # multiply
    num_layers = len(model_adapter.get_layers())
    if not os.path.exists(layer_cfg.cfg.tmp_path):
        raise ValueError('Path to orthogonal matrices does not exist. Provide correct path.')

    def get_Q(n, idx):
        return torch.load(os.path.join(layer_cfg.cfg.tmp_path, 'Qt{}_idx_{}.pt'.format(n, idx)))

    if args.init_from_saved:
        assign_attn_block(0, get_Q(1, 0), get_Q(2, 0), model_adapter, dtype, cfg, device)
        assign_ffn_block(0, get_Q(1, 0), get_Q(2, 0), model_adapter, dtype, cfg, device)
        for idx in range(1, num_layers):
            assign_attn_block(idx, get_Q(1, idx), get_Q(2, idx - 1), model_adapter, dtype, cfg, device)
            assign_ffn_block(idx, get_Q(1, idx), get_Q(2, idx), model_adapter, dtype, cfg, device)
        Q_head = torch.load(os.path.join(layer_cfg.cfg.tmp_path, 'Q_head.pt'))
        assign_head(-1, Q_head, get_Q(2, num_layers - 1), model_adapter, dtype, cfg, device, Q_emb=get_Q(1, 0))
        del Q_head, layer_cfg
        cleanup_memory()
        if log:
            print('end get_rotated_adapter', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
    return model_adapter, tokenizer, \
           {'original': original_param_count, 'after_fusion': after_fusion_param_count}


def init_model(args, model_adapter=None, tokenizer=None, params_num=None, log=False):
    dtype, device = process_args(args)
    with open(args.cfg_for_compression_path, 'r') as f:
        print(f.read(), flush=True)
    cfg_compr = OmegaConf.load(args.cfg_for_compression_path)
    layer_cfg = LayersConfig(OmegaConf.load(args.cfg_for_layers_path))
    if model_adapter is None:
        # load the pre-trained model
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model_name, args.model_path,
                                                                    token=args.hf_token, dtype=dtype)
        original_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
        print('Loaded adapter from hf', flush=True)
        # replace modules with compressible equivalents
        layernorm_fusion.replace_layers(model_adapter)
        # fuse layernorms and add rotations to skip connections
        layernorm_fusion.fuse_modules(model_adapter, detach_head=layer_cfg.cfg.detach_head)
        print('Fused modules', flush=True)
        after_fusion_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
        # add matrices to skip connections
        hidden_size = model_adapter.hidden_size
        for layer_adapter in model_adapter.get_layers():
            layer_adapter.layer.attn_shortcut_Q = nn.Parameter(torch.eye(hidden_size, device=device))
            layer_adapter.layer.mlp_shortcut_Q = nn.Parameter(torch.eye(hidden_size, device=device))
    else:
        assert tokenizer is not None
        if params_num is None:
            after_fusion_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
            original_param_count = after_fusion_param_count  # original param count may not be equal to this
        else:
            original_param_count = params_num['original']
            after_fusion_param_count = params_num['after_fusion']
    # model_adapter.model.to(config.device)
    print('Model name', args.model_name, flush=True)
    print(f'Original model parameters: {original_param_count:,d}', flush=True)
    print(f'After fusion model parameters: {after_fusion_param_count:,d}', flush=True)
    # get frequency of tokens
    dataset = data_utils.get_dataset(args.calibration_dataset)
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.seqlen,
        batch_size=args.batch_size,
        nsamples=cfg_compr.num_samples,
        varied_seqlen=False,
        seed=0,
    )
    cnt = 0
    frequency = {k: 0 for k in range(model_adapter.get_embeddings()[0].weight.shape[0])}
    for batch in train_loader:
        s = batch['input_ids'].numpy().reshape(-1)
        for k in s:
            if k in frequency.keys():
                frequency[k] += 1
            else:
                frequency[k] = 1
        cnt += batch['input_ids'].shape[0]
        if cnt > cfg_compr.num_samples:
            break
    frequency = torch.tensor([frequency[k] for k in range(0, len(frequency))]).float()
    # find optimal rotations
    if cfg_compr.norm == 'F':
        start = time.time()
        model_adapter = procrustes_init(model_adapter, cfg_compr, layer_cfg, frequency, device)
        if log:
            print('Init time', time.time() - start)
        if args.save_rotated:
            torch.save(model_adapter, os.path.join(args.save_path, 'rotated_' + args.filename))
        # replace layers
        if log:
            print('mem before replacement', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
        start = time.time()
        replace_all_layers(model_adapter.model, args.cfg_for_layers_path, args.model_name, init=True,
                           compress_head_emb=layer_cfg.cfg.detach_head, dtype=dtype)
        if log:
            print('mem after replacement', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, flush=True)
            print('replacement time', time.time() - start)
    elif cfg_compr.norm == 'W':
        weighted_procrustes_init(args.model_name, model_adapter, train_loader, layer_cfg, cfg_compr, frequency,
                                 device, cut_values=True, compress_emb=layer_cfg.cfg.detach_head)
    else:
        raise NotImplementedError()
    if args.save:
        torch.save(model_adapter, os.path.join(args.save_path, args.filename))
    new_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
    print(f'After replacement model parameters: {new_param_count:,d}')
    return model_adapter, tokenizer, \
           {'original': original_param_count, 'after_fusion': after_fusion_param_count, 'new': new_param_count}


@torch.no_grad()
def evaluate_model(args, model_adapter, tokenizer):
    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(torch.device('cuda'))

    dataset = data_utils.get_dataset(args.ppl_eval_dataset)
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size)
    # reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    print(f'Ppl {dataset_ppl:.4f}')
    return dataset_ppl
