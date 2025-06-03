import os
import time
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from procrustes.compression_pipeline import procrustes_arg_parser, evaluate_model, init_model, get_rotated_adapter
from procrustes.skip_connection import process_skip_connections


def main():
    args = procrustes_arg_parser(interactive=True)
    cfg = OmegaConf.load(args.cfg_for_compression_path)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    try:
        mp.set_sharing_strategy('file_system')
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print('error occured in multiprocessing')
        return None
    if cfg.norm == 'F':
        model_adapter, tokenizer, params_num = None, None, None
    else:
        model_adapter, tokenizer, params_num = get_rotated_adapter(args)

    model_adapter, tokenizer, params_num = init_model(args, model_adapter, tokenizer, params_num)

    if args.skip_connections is not None:
        process_skip_connections(model_adapter, args.skip_connections)
        print('Processed skip connections')
    else:
        print('Skip connections are not compressed')

    model_adapter.model.half()
    model_adapter.model.cuda()
    original_param_count = params_num['original']
    after_fusion_param_count = params_num['after_fusion']
    new_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
    print(f'New model parameters: {new_param_count:,d}', flush=True)
    fraction = 1.0 - new_param_count / original_param_count
    print(f'From original fraction {fraction:.4f}')
    fraction = 1.0 - new_param_count / after_fusion_param_count
    print(f'From fused fraction {fraction:.4f}')

    ppl = evaluate_model(args, model_adapter, tokenizer)
    print('Ppl', ppl)


if __name__ == '__main__':
    main()
