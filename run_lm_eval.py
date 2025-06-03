# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os

import lm_eval
import torch
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from transformers import AutoTokenizer

from slicegpt import gpu_utils, utils
from slicegpt.config import config

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}


def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    global ALL_TASKS
    manager = TaskManager()
    ALL_TASKS = manager.initialize_tasks()
    print('initialized')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to load the model from",
    )
    path_group.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to load the tokenizer from",
        default="facebook/opt-125m",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"]
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the lm eval results")
    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    return parser.parse_args() if interactive else parser.parse_args('')


def process_eval_args(args: argparse.Namespace):
    logging.info(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.info(f'{arg} = {argv}')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())
    wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


def eval_main(args: argparse.Namespace) -> None:
    logging.info("Running LM eval experiment.")

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    logging.info(f"Loading model from {args.weights_path}")
    model_adapter = torch.load(args.weights_path, weights_only=False, map_location=torch.device('cpu'))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, token=None) #local_files_only=True)
    print('loaded model_adapter and tokenizer')

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model_adapter.model.tie_weights = lambda: None

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    ### LM Eval Harness ###
    hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=args.batch_size)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    print('task_names', args.tasks, task_names)

    logging.info(f"Selected Tasks: {task_names}")

    for task in task_names:
        if task not in TASK_METRIC_MAP:
            raise NotImplementedError(
                f"Please specify the metric to use for {task} in TASK_METRIC_MAP. Available info {TASK_METRIC_MAP}"
            )

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)[
        'results'
    ]

    logging.info(results)
    if not args.no_wandb:
        wandb.log(results)
    print(results)
    for name in results.keys():
        if 'acc_norm_stderr,none' in results[name]:
            print(results[name]['alias'], results[name]['acc,none'], results[name]['acc_norm_stderr,none'])
        elif 'acc_stderr,none' in results[name]:
            print(results[name]['alias'], results[name]['acc,none'], results[name]['acc_stderr,none'])

    with open(f"{args.save_dir}/full_results_{args.num_fewshot}_shot.json", "w") as f:
        json.dump(results, f)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)
    with open(f"{args.save_dir}/{args.num_fewshot}_shot_task_results.json", "w") as f:
        json.dump(metric_vals, f)

    if not args.no_wandb:
        wandb.log({'acc_avg': acc_avg})
    print('acc_avg', acc_avg)

    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logging = lm_eval_utils.eval_logger
    logging.addHandler(utils.create_file_handler(log_dir="log"))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    eval_args = eval_arg_parser()
    process_eval_args(eval_args)
    eval_main(eval_args)