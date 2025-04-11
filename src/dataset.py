import pickle
import sys, os
from functools import partial

import pandas as pd
import random
from tqdm.auto import tqdm
import numpy as np

import ipdb
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("src"))
from utils import read_pickle, write_pickle


class DescriptionDataset(Dataset):
    def __init__(self, code_descriptions, tokenizer):
        self.definitions = code_descriptions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.definitions)

    def __getitem__(self, idx):
        code_description = self.definitions[idx]
        result = self.tokenizer(
            code_description,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "idx": idx,
            "input_ids": result["input_ids"][0],
            "attention_mask": result["attention_mask"][0],
        }


class SynonymDataset(Dataset):
    def __init__(self, idx2code, code_descriptions, code_synonyms, tokenizer):
        self.idx2code = idx2code
        self.descriptions = code_descriptions
        self.code2synonym = code_synonyms
        self.total_synonyms = self.prepare_synonyms()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.descriptions)

    def prepare_synonyms(self):
        all_synonyms = []
        for i in range(len(self.idx2code)):
            code = self.idx2code[i]
            synonyms = self.code2synonym[code]
            if len(synonyms) == 0:
                definition = self.descriptions[i]
                synonyms.append(definition)
            if len(synonyms) > 4:
                synonyms = synonyms[:4]
            elif len(synonyms) == 4:
                pass
            elif len(synonyms) == 3:
                synonyms.append(synonyms[0])
            elif len(synonyms) == 2:
                synonyms.extend([synonyms[0], synonyms[1]])
            elif len(synonyms) == 1:
                synonyms.extend([synonyms[0], synonyms[0], synonyms[0]])
            all_synonyms.extend(synonyms)
        return all_synonyms

    def __getitem__(self, idx):
        synonym = self.total_synonyms[idx]

        result = self.tokenizer(
            synonym,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "idx": idx,
            "input_ids": result["input_ids"][0],
            "attention_mask": result["attention_mask"][0],
        }


class EHRDataset(Dataset):

    def __init__(
        self,
        pkl_file_path,
        tokenizer,
        max_length,
        chunk_size,
        code2idx,
        use_guidance=False,
        code_descriptions=None,
        use_shuffle=False,
        use_synonyms=False,
        code_synonyms=None,
        use_hierarchy=False,
        code_hierarchy=None,
    ):
        if pkl_file_path.endswith(".pkl"):
            self.samples = read_pickle(pkl_file_path)
        else:
            raise ValueError("Invalid file format")

        self.tokenizer, self.max_length, self.chunk_size = (
            tokenizer,
            max_length,
            chunk_size,
        )
        self.code2idx = code2idx
        self.use_guidance = use_guidance
        self.use_shuffle = use_shuffle
        self.use_synonyms = use_synonyms
        self.use_hierarchy = use_hierarchy
        self.code_descriptions = code_descriptions
        if use_guidance and use_synonyms:
            self.code_synonyms = {}
            # ensure that each code has at least one synonym
            for code in code2idx:
                idx = self.code2idx[code]
                definition = self.code_descriptions[idx]
                if code not in code_synonyms:
                    self.code_synonyms[code] = [definition]
                else:
                    synonyms = code_synonyms[code]
                    if len(synonyms) == 0:
                        synonyms.append(definition)
                    self.code_synonyms[code] = synonyms
        if use_guidance and use_hierarchy:
            self.code_hierarchy: dict[str:str] = code_hierarchy

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dic = {}
        dic["idx"] = idx
        dic["raw_text"] = sample["text"]
        dic["raw_label"] = sample["label"]

        (
            dic["input_ids"],
            dic["attention_mask"],
            dic["num_chunks"],
            dic["offset_mapping"],
        ) = self.preprocess_text(sample["text"])
        dic["label"] = self.get_onehot_label(sample["label"])
        if self.use_guidance:
            (
                dic["guidance_input_ids"],
                dic["guidance_attention_mask"],
                dic["guidance_num_chunks"],
                label_offset_mapping,
            ) = self.get_synthetic_sample(sample["label"], self.use_synonyms)
        return dic

    def preprocess_text(self, text):
        """
        text: str
        return: input_ids, attention_mask, num_chunks
            input_ids: (num_chunks, chunk_size)
            attention_mask: (num_chunks, chunk_size)
            num_chunks: int
        """
        chunk_size = self.chunk_size
        result = self.tokenizer(
            text,
            padding=False,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        input_ids, attention_mask = result["input_ids"], result["attention_mask"]
        offset_mapping = result["offset_mapping"]

        num_token = len(input_ids)
        if num_token % chunk_size == 0:
            num_chunk = num_token // chunk_size
        else:
            num_chunk = num_token // chunk_size + 1
        num_chunk = min(num_chunk, self.max_length // chunk_size)
        max_length = num_chunk * chunk_size
        pad_length = max_length - num_token
        input_ids = (
            input_ids + [1] * pad_length
        )  # 1 means padding in roberta's vocabulary
        attention_mask = attention_mask + [0] * pad_length
        return (
            torch.tensor(input_ids).view(num_chunk, chunk_size),
            torch.tensor(attention_mask).view(num_chunk, chunk_size),
            num_chunk,
            offset_mapping,
        )

    def get_onehot_label(self, label):
        """
        label: list of str
        return: onehot_label
        """
        onehot_label = torch.zeros(len(self.code2idx))
        # binary label
        for code in label:
            if code != "":
                # code as label
                idx = self.code2idx[code]
                onehot_label[idx] = 1

        return onehot_label

    def get_synthetic_sample(self, codes, use_synonyms=False, use_hierarchy=False):
        code_ids = [self.code2idx[code] for code in codes]
        if self.use_shuffle:
            random.shuffle(code_ids)
        if use_synonyms:
            code_expressions = [
                random.choice(self.code_synonyms[code]) for code in codes
            ]
        else:
            code_expressions = [self.code_descriptions[code_id] for code_id in code_ids]
        if use_hierarchy:
            code_hierarchy = [self.code_hierarchy[code_id] for code_id in code_ids]
            code_expressions = [
                hierarchy + ", " + code_expression
                for code_expression, hierarchy in zip(code_expressions, code_hierarchy)
            ]
        synthetic_sample = ". ".join(code_expressions)
        return self.preprocess_text(synthetic_sample)


def data_collator(
    samples,
):
    batch = dict()

    # batch["idx"] = [sample["idx"] for sample in samples]

    for key in [
        "input_ids",
        "attention_mask",
        "guidance_input_ids",
        "guidance_attention_mask",
    ]:
        if key in samples[0]:
            batch[key] = torch.cat(
                [sample[key] for sample in samples]
            )  # (num_chunks, chunk_size)

    for key in ["num_chunks", "guidance_num_chunks"]:
        if key in samples[0]:
            batch[key] = [sample[key] for sample in samples]

    # one-hot labels
    for key in ["label"]:
        if key in samples[0]:
            # (batch_size, num_labels)
            batch[key] = torch.cat([sample[key].unsqueeze(0) for sample in samples])

    return batch


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "models/RoBERTa-base-PM-M3-Voc-distill-align-hf",
        use_fast=True,
        do_lower_case=True,
    )
    dev_data_path = "data/mimic4_icd9/train.pkl"
    code_df = pd.read_csv("data/mimic4_icd9/codes.csv", sep="\t")
    code_list = code_df["code"].tolist()
    code_descriptions = code_df["desc"].tolist()
    code2idx = {code: idx for idx, code in enumerate(code_list)}
    dev_dataset = EHRDataset(dev_data_path, tokenizer, 3072, 128, code2idx)

    batch_size = 2
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    for batch in tqdm(dev_dataloader):
        pass
