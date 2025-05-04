import argparse
import pickle
import json


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(_, path):
    with open(path, "wb") as f:
        pickle.dump(_, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--code_description_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help=("The size of chunks that we'll split the inputs into"),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--cased",
        action="store_true",
        help="equivalent to do_lower_case=False",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=0,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--code_50", action="store_true", help="use only top-50 codes")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--embed_code_query",
        action="store_true",
    )
    parser.add_argument(
        "--freeze_code_query",
        action="store_true",
    )
    parser.add_argument(
        "--use_guidance",
        action="store_true",
    )
    parser.add_argument(
        "--use_synonyms",
        action="store_true",
    )
    parser.add_argument(
        "--code_synonyms_file",
        type=str,
    )
    parser.add_argument(
        "--use_shuffle",
        action="store_true",
    )
    parser.add_argument(
        "--use_sim_loss",
        action="store_true",
    )
    parser.add_argument(
        "--lambda_sim_loss",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--code_distribution_file",
        type=str,
    )
    parser.add_argument(
        "--extra_code_description_file",
        type=str,
    )
    parser.add_argument(
        "--use_hierarchy",
        action="store_true",
    )
    parser.add_argument(
        "--code_group_file",
        type=str,
    )
    parser.add_argument(
        "--code_relation_file",
        type=str,
    )
    parser.add_argument(
        "--use_cross_attention",
        action="store_true",
    )
    parser.add_argument(
        "--use_biaffine",
        action="store_true",
    )
    parser.add_argument(
        "--use_rdrop",
        action="store_true",
    )
    parser.add_argument(
        "--rdrop_alpha",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--use_2stage",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
    )
    parser.add_argument(
        "--find_best_threshold",
        action="store_true",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--use_swanlab",
        action="store_true",
    )
    parser.add_argument(
        "--save_pred_results",
        action="store_true",
    )
    parser.add_argument(
        "--save_topk_results",
        action="store_true",
    )
    parser.add_argument(
        "--save_group_results",
        action="store_true",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    return args
