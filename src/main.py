import logging
import math
import os
import random
import pickle
import json
from functools import partial

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from accelerate import Accelerator, DistributedDataParallelKwargs
from swanlab.integration.accelerate import SwanLabTracker
import transformers
import datasets
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    get_scheduler,
    set_seed,
)

from dataset import EHRDataset, data_collator, DescriptionDataset
from model import IcdCodeModel, Output
from utils import parse_args
from evaluation import (
    metric_with_threshold,
    metric_without_threshold,
    multilabel_ranking_average_precision,
    get_code_group_mask,
    metric_by_group,
)


def limit_threads(cpu_num):
    cpu_num = 4
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


logger = logging.getLogger(__name__)


def get_best_threshold(all_preds, all_labels):
    all_preds = all_preds.flatten()
    all_labels = all_labels.flatten()
    sort_arg = torch.argsort(all_preds)
    sort_label = torch.take(all_labels, sort_arg)
    label_count = torch.sum(sort_label)
    correct = label_count - torch.cumsum(sort_label, dim=0)
    predict = all_labels.shape[0] + 1 - torch.cumsum(torch.ones_like(sort_label), dim=0)
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = torch.take(all_preds, sort_arg)
    f1_argmax = torch.argmax(f1)
    best_threshold = sort_yhat_raw[f1_argmax]
    return best_threshold.item()


def write_metrics(sub, metrics, output_dir, filename):
    metric_names, metric_values = metrics.keys(), metrics.values()
    metric_names = [""] + [k for k in metric_names]
    metric_values = [sub] + ["{:.4f}".format(v) for v in metric_values]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}/{filename}"):
        with open(f"{output_dir}/{filename}", "w") as f:
            f.write(",".join(metric_names) + "\n")
            f.write(",".join(metric_values) + "\n")
    else:
        with open(f"{output_dir}/{filename}", "a") as f:
            f.write(",".join(metric_values) + "\n")


def save_pred_results(
    all_preds,
    all_labels,
    test_dataset,
    code2idx,
    idx2code,
    code_descriptions,
    output_dir,
    save_topk_results=False,
    all_topk_logits=None,
    all_topk_ids=None,
    all_att_value=None,
    all_att_pos=None,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if save_topk_results:
        all_topk_logits = all_topk_logits.tolist()
        all_topk_ids = all_topk_ids.tolist()
        all_att_value = all_att_value.tolist()
        all_att_pos = all_att_pos.tolist()

    for i in range(len(all_preds)):
        sample = test_dataset[i]
        raw_text = sample["raw_text"]
        pred_logits = all_preds[i]
        pred_idx = (pred_logits > 0.5).nonzero(as_tuple=True)[0].tolist()
        logits_ = pred_logits[pred_idx]
        pred_codes = [idx2code[idx] for idx in pred_idx]
        label_codes = sample["raw_label"]
        label_idx = [code2idx[code] for code in label_codes]
        label_pred = [round(pred_logits[_].item(), 4) for _ in label_idx]
        recall = (
            len(set(label_codes) & set(pred_codes)) / len(label_codes)
            if len(label_codes) > 0
            else 0
        )
        precision = (
            len(set(label_codes) & set(pred_codes)) / len(pred_codes)
            if len(pred_codes) > 0
            else 0
        )
        f1 = (
            round(2 * recall * precision / (recall + precision), 4)
            if recall + precision > 0
            else 0
        )

        if save_topk_results:
            topk_idx, topk_logits, att_pos, att_value = (
                all_topk_ids[i],
                all_topk_logits[i],
                all_att_pos[i],
                all_att_value[i],
            )
            topk_codes = [idx2code[idx] for idx in topk_idx]
            offset_mapping = sample["offset_mapping"]
            label_att_pos = [att_pos[idx] for idx in label_idx]
            label_att_value = [round(att_value[idx], 4) for idx in label_idx]
            topk_att_pos = [att_pos[idx] for idx in topk_idx]
            topk_att_value = [round(att_value[idx], 4) for idx in topk_idx]
            topk_pred = [round(v, 4) for v in topk_logits]

        with open(os.path.join(output_dir, f"{i}_{f1}.txt"), "w") as f:
            f.write(f"Text: {raw_text}\n")
            f.write(f"Labels:\n")
            for code, pred in zip(label_codes, label_pred):
                f.write(f"{code} {code_descriptions[code2idx[code]]} {pred}\n")
            f.write(f"Pred Codes:\n")
            for code, pred in zip(pred_codes, logits_):
                f.write(f"{code} {code_descriptions[code2idx[code]]}\n")
            f.write(f"F1: {f1}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")

            if save_topk_results:
                f.write(f"Topk:\n")
                for code, pos, value, pred in zip(
                    topk_codes, topk_att_pos, topk_att_value, topk_pred
                ):
                    if 0 < pos < len(offset_mapping) - 1:
                        s, e = offset_mapping[pos]
                        corresponding_text = raw_text[s:e]
                    else:
                        corresponding_text = "padding"
                    f.write(
                        f"{code} {code_descriptions[code2idx[code]]}\n  Att: {corresponding_text} {value} Pred: {pred}\n"
                    )


def main():
    args = parse_args()

    limit_threads(args.num_cores)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    if args.use_swanlab:
        tracker = SwanLabTracker(
            project=args.dataset,
            experiment_name=args.name,
        )
        accelerator = Accelerator(
            log_with=tracker,
            kwargs_handlers=[ddp_kwargs],
        )
        accelerator.init_trackers(
            project_name=args.dataset,
            config={
                "batch_size": args.per_device_train_batch_size
                * args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "chunk_size": args.chunk_size,
                "use_cross_attention": args.use_cross_attention,
                "embed_code_query": args.embed_code_query,
                "use_guidance": args.use_guidance,
                "use_shuffle": args.use_shuffle,
                "use_synonyms": args.use_synonyms,
                "use_hierarchy": args.use_hierarchy,
                "use_sim_loss": args.use_sim_loss,
                "lambda_sim_loss": args.lambda_sim_loss,
                "use_rdrop": args.use_rdrop,
                "rdrop_alpha": args.rdrop_alpha,
                "use_biaffine": args.use_biaffine,
            },
        )

    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Load pretrained model and tokenizer
    #
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        do_lower_case=not args.cased,
    )

    # Code Description & Synonyms
    config.code_description_file = args.code_description_file
    code_df = pd.read_csv(args.code_description_file, sep="\t")
    if args.use_synonyms:
        config.code_synonyms_file = args.code_synonyms_file
        with open(args.code_synonyms_file, "r") as f:
            code_synonyms = json.load(f)
    else:
        code_synonyms = None
    code_list = code_df["code"].tolist()
    code_descriptions = code_df["desc"].tolist()
    code2idx = {code: idx for idx, code in enumerate(code_list)}
    idx2code = {idx: code for idx, code in enumerate(code_list)}

    # Code & Frequency
    code_distribution_df = pd.read_csv(args.code_distribution_file, sep=",")
    code2freq = dict(
        zip(
            code_distribution_df["code"].tolist(),
            code_distribution_df["total"].tolist(),
        )
    )
    group2mask = get_code_group_mask(code2idx, code2freq)

    # Hierarchical Relationship Guidance
    if args.use_hierarchy:
        code_hierarchy = dict()
        group_df = pd.read_csv(args.code_group_file, sep="\t")
        group2name = {
            group: name for group, name in zip(group_df["group"], group_df["name"])
        }
        relation_df = pd.read_csv(args.code_relation_file, sep="\t")
        extra_code_df = pd.read_csv(args.extra_code_description_file, sep="\t")
        code2desc = {
            code: name
            for code, name in zip(extra_code_df["code"], extra_code_df["desc"])
        }
        code2desc.update(dict(zip(code_list, code_descriptions)))
        for idx, row in relation_df.iterrows():
            code = row["code"]
            code_before = code.split(".")[0]
            # Group information
            if len(code_before) == 2:
                proc = row["proc"]
                template = group2name[proc]
            else:
                diag1, diag2 = row["diag1"], row["diag2"]
                template = group2name[diag1] + ", " + group2name[diag2]
            # Extra Code Information
            if code != code_before:
                template = template + ", " + code2desc[code_before]
            # if len(code) - len(code_before) > 2:
            #     template = template + ", " + code_before_to_name[code[:-1]]
            code_hierarchy[code] = template
    else:
        code_hierarchy = None
        # print(code_hierarchy)
        # exit()
    # Code Embedding Initilization Using Code Definition
    if args.embed_code_query:
        definition_dataset = DescriptionDataset(
            code_descriptions=code_descriptions, tokenizer=tokenizer
        )
        description_dataloader = DataLoader(
            definition_dataset,
            pin_memory=True,
            batch_size=64,
        )

    # Pretrained Model
    config.model_name_or_path = args.model_name_or_path
    # num_labels
    config.num_labels = len(code_list)
    # num of retrieve embeddings
    config.num_embeddings = len(code_list)
    # Architecture
    config.use_cross_attention = args.use_cross_attention
    # Use Synthetic Data As Extra Samples to guide the model
    config.use_guidance = args.use_guidance
    config.use_synonyms = args.use_synonyms
    config.use_hierarchy = args.use_hierarchy
    config.use_sim_loss = args.use_sim_loss
    config.lambda_sim_loss = args.lambda_sim_loss

    # Tricks
    config.use_biaffine = args.use_biaffine
    config.use_rdrop = args.use_rdrop
    config.rdrop_alpha = args.rdrop_alpha

    my_collate_fn = partial(data_collator)

    if args.num_train_epochs > 0:
        train_dataset = EHRDataset(
            args.train_file,
            tokenizer,
            args.max_length,
            args.chunk_size,
            code2idx,
            use_guidance=args.use_guidance,
            code_descriptions=code_descriptions,
            use_shuffle=args.use_shuffle,
            use_synonyms=args.use_synonyms,
            code_synonyms=code_synonyms,
            use_hierarchy=args.use_hierarchy,
            code_hierarchy=code_hierarchy,
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=my_collate_fn,
            pin_memory=True,
            batch_size=args.per_device_train_batch_size,
        )

        dev_dataset = EHRDataset(
            args.validation_file,
            tokenizer,
            args.max_length,
            args.chunk_size,
            code2idx,
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            collate_fn=my_collate_fn,
            pin_memory=True,
            batch_size=args.per_device_eval_batch_size,
        )

    test_dataset = EHRDataset(
        args.test_file,
        tokenizer,
        args.max_length,
        args.chunk_size,
        code2idx,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=my_collate_fn,
        pin_memory=True,
        batch_size=args.per_device_eval_batch_size,
    )

    if args.num_train_epochs > 0:
        model: IcdCodeModel = IcdCodeModel(config, load_plm_weights=True)
    else:
        model: IcdCodeModel = IcdCodeModel.from_pretrained(
            args.output_dir,
            config=config,
        )

    # Initialzation w/ ICD Knowledge
    if args.num_train_epochs > 0 and args.embed_code_query:
        model.eval()
        model.to(torch.device("cuda:0"))
        for batch in description_dataloader:
            model.retrieve_embed_init(**batch)
        model.classifier_embed_init()

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    if args.num_train_epochs > 0:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        train_dataloader, dev_dataloader, optimizer = accelerator.prepare(
            train_dataloader, dev_dataloader, optimizer
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=num_update_steps_per_epoch * args.num_train_epochs,
        )

    if args.num_train_epochs > 0:
        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        # Training
        for epoch in tqdm(range(args.num_train_epochs)):
            model.train()
            epoch_loss = 0.0
            batch_group = []
            for step, batch in enumerate(train_dataloader):
                # gradient accumulation
                batch_group.append(batch)
                num_batch = len(batch_group)
                if (
                    num_batch < args.gradient_accumulation_steps
                    and step < len(train_dataloader) - 1
                ):
                    continue
                bce_loss, sim_loss, rdrop_loss, map, guidance_map = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                for batch in batch_group:
                    outputs: Output = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss / num_batch)
                    bce_loss += outputs.bce_loss.item() / num_batch
                    batch_size, num_labels = batch["label"].size(0), batch[
                        "label"
                    ].size(1)
                    map += (
                        multilabel_ranking_average_precision(
                            outputs.logits.detach()[:batch_size],
                            batch["label"].detach(),
                            num_labels=num_labels,
                        ).item()
                        / num_batch
                    )
                    if args.use_guidance:
                        guidance_map += (
                            multilabel_ranking_average_precision(
                                outputs.logits.detach()[batch_size : batch_size * 2],
                                batch["label"].detach(),
                                num_labels=num_labels,
                            ).item()
                            / num_batch
                        )
                        if args.use_sim_loss:
                            sim_loss += outputs.sim_loss.item() / num_batch

                    if args.use_rdrop:
                        rdrop_loss += outputs.rdrop_loss.item() / num_batch

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                batch_group = []
                epoch_loss += bce_loss
                progress_bar.set_postfix(
                    {
                        "bce_loss": bce_loss,
                        "map": map,
                        "guideline_map": guidance_map,
                        "sim_loss": sim_loss,
                        "rdrop_loss": rdrop_loss,
                    }
                )
                if args.use_swanlab:
                    accelerator.log(
                        {
                            "bce_loss": bce_loss,
                            "map": map,
                            "guideline_map": guidance_map,
                            "sim_loss": sim_loss,
                            "rdrop_loss": rdrop_loss,
                        },
                        step=completed_steps,
                    )

            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            for step, batch in tqdm(enumerate(dev_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                preds = outputs.logits.sigmoid()
                all_preds.append(preds)
                all_labels.append(batch["label"])
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = metric_with_threshold(all_preds, all_labels, threshold=0.5)
            metrics.update(
                metric_without_threshold(all_preds, all_labels, k_list=[5, 8, 15])
            )
            logger.info(f"epoch {epoch} finished")
            logger.info(f"dev dataset metrics: {metrics}")
            metrics = {"dev" + k: v for k, v in metrics.items()}
            metrics.update({"bce_loss_epoch": epoch_loss / num_update_steps_per_epoch})
            if args.use_swanlab:
                accelerator.log(metrics, step=epoch)
            write_metrics(
                "epoch={}".format(epoch), metrics, args.output_dir, "train_metrics.csv"
            )

    # Test
    if accelerator.is_local_main_process:
        model.eval()
        all_preds = []
        all_labels = []
        all_topk_logits, all_topk_ids = [], []
        all_att_pos, all_att_value = [], []
        for step, batch in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
                # (bs, k)
            preds = outputs.logits.sigmoid()
            all_preds.append(preds)
            all_labels.append(batch["label"])
            if args.save_topk_results:
                # topk
                # (bs,k)
                topk_pred, topk_idx = model.get_topk_pred(outputs.logits)
                all_topk_logits.append(topk_pred)
                all_topk_ids.append(topk_idx)
                att_value, att_pos = model.get_max_att_weight_and_pos(
                    outputs.attention_weights
                )
                all_att_value.append(att_value)
                all_att_pos.append(att_pos)
            # break
        # (num_samples, num_labels)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        if args.save_topk_results:
            # topk: (num_samples, k)
            all_topk_logits = torch.cat(all_topk_logits)
            all_topk_ids = torch.cat(all_topk_ids)
            all_att_pos = torch.cat(all_att_pos)
            all_att_value = torch.cat(all_att_value)

        _ = metric_without_threshold(all_preds, all_labels, k_list=[5, 8, 15])
        logger.info(f"evaluation finished")
        logger.info(f"metrics: {_}")
        # get best threshold
        if args.find_best_threshold:
            best_threshold = round(get_best_threshold(all_preds, all_labels), 2)
            logger.info(f"best threshold: {best_threshold}")
            metrics = metric_with_threshold(
                all_preds, all_labels, threshold=best_threshold
            )
            logger.info(f"metrics for best threshold: {metrics}")
            metrics.update(_)
            write_metrics(
                "best_threshold={}".format(best_threshold),
                metrics,
                args.output_dir,
                "test_metrics.csv",
            )
            metrics = {"test" + k: v for k, v in metrics.items()}
            if args.use_swanlab:
                accelerator.log(metrics, step=0)

        thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for i, t in enumerate(thresholds):
            metrics = metric_with_threshold(all_preds, all_labels, threshold=t)
            logger.info(f"metrics for threshold {t}: {metrics}")
            metrics.update(_)
            write_metrics(
                "threshold={}".format(t), metrics, args.output_dir, "test_metrics.csv"
            )
            metrics = {"test" + k: v for k, v in metrics.items()}

            if args.use_swanlab:
                accelerator.log(metrics, step=i)

        if args.save_group_results:
            # Metric by group
            """
            >500
            101-500
            51-100
            11-50
            1-10
            """
            group2metric = metric_by_group(
                all_preds, all_labels, group2mask, best_threshold
            )

            with open(f"{args.output_dir}/group_metrics.txt", "w") as f:
                for i, (group, metrics) in enumerate(group2metric.items()):
                    f1, precision, recall = (
                        metrics["f1"],
                        metrics["precision"],
                        metrics["recall"],
                    )
                    f.write(
                        f"Codes: {group}, f1: {f1}, precision: {precision}, recall: {recall}\n"
                    )
                    print(
                        f"Codes: {group}, f1: {f1}, precision: {precision}, recall: {recall}"
                    )
                    if args.use_swanlab:
                        accelerator.log(
                            {
                                f"test_f1_by_freq": f1,
                                f"test_prec_by_freq": precision,
                                f"test_recall_by_freq": recall,
                            },
                            step=i,
                        )

        # Save the model if training
        if args.output_dir is not None and args.num_train_epochs > 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, save_function=accelerator.save
            )
            accelerator.end_training()

        # for Case Study
        if args.save_pred_results:
            save_pred_results(
                all_preds,
                all_labels,
                test_dataset,
                code2idx,
                idx2code,
                code_descriptions,
                output_dir=f"{args.output_dir}/results/",
                save_topk_results=args.save_topk_results,
                all_topk_logits=all_topk_logits,
                all_topk_ids=all_topk_ids,
                all_att_value=all_att_value,
                all_att_pos=all_att_pos,
            )


if __name__ == "__main__":
    main()
