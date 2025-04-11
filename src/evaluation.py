import torch
from torchmetrics.functional import (
    accuracy,
    f1_score,
    precision,
    recall,
    auroc,
)
from torchmetrics.functional.classification import multilabel_ranking_average_precision
import numpy as np


def metric_with_threshold(pred, target, threshold=0.5):
    device = pred.device
    target = target.to(device)
    num_all_codes = target.size(1)
    metrics = dict()

    # macro (only calculate for existing codes)
    existing_code_mask = target.sum(0) > 0
    num_existing_codes = existing_code_mask.sum().item()
    metrics["f1_macro_revised"] = f1_score(
        pred[:, existing_code_mask],
        target[:, existing_code_mask],
        task="multilabel",
        threshold=threshold,
        average="macro",
        num_labels=num_existing_codes,
    )
    # marco (calculate for all codes)
    metrics["f1_macro"] = f1_score(
        pred,
        target,
        task="multilabel",
        threshold=threshold,
        average="macro",
        num_labels=num_all_codes,
    )

    # micro
    metrics["f1_micro"] = f1_score(
        pred,
        target,
        task="multilabel",
        threshold=threshold,
        average="micro",
        num_labels=num_all_codes,
    )

    metrics = {k: v.item() for k, v in metrics.items()}
    return metrics


def precision_recall_at_k(pred_at_k, target):
    # pred_at_k: (bs, num_labels)
    # target: (bs, num_labels)
    tp = (pred_at_k * target).sum(1)
    pred_sum = pred_at_k.sum(1)
    traget_sum = target.sum(1)
    precision_at_k, recall_at_k = torch.mean(tp / pred_sum), torch.mean(tp / traget_sum)
    return precision_at_k, recall_at_k


def metric_without_threshold(pred, target, k_list=[5, 8, 15]):
    device = pred.device
    target = target.long().to(device)
    num_labels = target.size(1)
    existing_code_mask = target.sum(0) > 0
    num_existing_codes = existing_code_mask.sum().item()
    metrics = dict()

    # mean average precision
    metrics["map"] = multilabel_ranking_average_precision(
        pred, target, num_labels=num_labels
    )

    # precision at k
    for k in k_list:
        topk_ids = pred.topk(k, dim=1)[1]
        pred_topk = torch.zeros_like(pred).to(device)
        pred_topk = pred_topk.scatter(1, topk_ids, 1)
        metrics[f"prec_at_{k}"], _ = precision_recall_at_k(pred_topk, target)

    # auroc
    metrics["auc_macro"] = auroc(
        pred[:, existing_code_mask].cpu(),
        target[:, existing_code_mask].cpu(),
        task="multilabel",
        average="macro",
        num_labels=num_existing_codes,
    )
    metrics["auc_micro"] = auroc(
        pred.cpu(),
        target.cpu(),
        task="multilabel",
        average="micro",
        num_labels=num_labels,
    )
    metrics = {k: v.item() for k, v in metrics.items()}

    return metrics


def get_code_group_mask(code2idx, code2frequency):
    code2group = {}
    group2mask = {
        ">500": [0] * len(code2idx),
        "101-500": [0] * len(code2idx),
        "51-100": [0] * len(code2idx),
        "11-50": [0] * len(code2idx),
        "1-10": [0] * len(code2idx),
    }
    for code, idx in code2idx.items():
        code_frequency = code2frequency.get(code, 0)
        if code_frequency > 500:
            code_group = ">500"
        elif code_frequency > 100:
            code_group = "101-500"
        elif code_frequency > 50:
            code_group = "51-100"
        elif code_frequency > 10:
            code_group = "11-50"
        else:
            code_group = "1-10"
        code2group[code] = code_group
        group2mask[code_group][idx] = 1
    group2mask = {k: torch.tensor(v) for k, v in group2mask.items()}
    return group2mask


def metric_by_group(pred, gt, group2mask, threshold=0.5):

    group2metrics = {}
    num_codes = pred.size(1)
    for code_group in group2mask.keys():
        mask = group2mask[code_group].to(pred.device)
        pred_group = pred * mask
        gt_group = gt * mask
        f1 = f1_score(
            pred_group,
            gt_group,
            task="multilabel",
            threshold=threshold,
            average="micro",
            num_labels=num_codes,
        )
        p = precision(
            pred_group,
            gt_group,
            task="multilabel",
            threshold=threshold,
            average="micro",
            num_labels=num_codes,
        )
        r = recall(
            pred_group,
            gt_group,
            task="multilabel",
            threshold=threshold,
            average="micro",
            num_labels=num_codes,
        )
        group2metrics[code_group] = {
            "f1": round(f1.item(), 4),
            "precision": round(p.item(), 4),
            "recall": round(r.item(), 4),
        }

    return group2metrics


if __name__ == "__main__":

    pred = torch.rand((10, 100))
    target = torch.randint(0, 2, (10, 100))
    existing_code_mask = target.sum(0) > 0
    num_existing_codes = existing_code_mask.sum().item()
    print(
        auroc(
            pred[:, existing_code_mask].cpu(),
            target[:, existing_code_mask].cpu(),
            task="multilabel",
            average="macro",
            num_labels=num_existing_codes,
        )
    )
    print(metric_with_threshold(pred, target))
