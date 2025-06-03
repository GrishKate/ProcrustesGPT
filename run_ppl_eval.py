import os
import torch
import argparse
from procrustes.compression_pipeline import evaluate_model, process_args
from transformers import AutoTokenizer


def arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-125m",
        help="Huggingface model name",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to load the tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to load the saved model from",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument('--device', type=str, default=None,
                        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.", )
    parser.add_argument("--ppl_eval_dataset", type=str, help="Calibration dataset", choices=["wikitext2"],
                        default="wikitext2")
    parser.add_argument("--ppl_eval_batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--distribute-model", action="store_true",
                        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.", )
    return parser.parse_args() if interactive else parser.parse_args('')


def main(args):
    dtype, device = process_args(args)
    tokenizer_path = args.tokenizer_path if args.tokenizer_path is not None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, token=args.hf_token,
                                              local_files_only=args.tokenizer_path is not None)
    model_adapter = torch.load(args.weights_path, weights_only=False, map_location=torch.device('cpu'))
    model_adapter.model.to(dtype)
    model_adapter.model.to(device)
    res = evaluate_model(args, model_adapter, tokenizer)
    print('Ppl', res)


if __name__ == '__main__':
    args = arg_parser()
    main(args)
