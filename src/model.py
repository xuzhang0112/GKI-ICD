from typing import Optional
from dataclasses import dataclass


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModel, PreTrainedModel, RobertaModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class Output(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    bce_loss: Optional[torch.FloatTensor] = None
    sim_loss: Optional[torch.FloatTensor] = None
    rdrop_loss: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None


class IcdCodeModel(PreTrainedModel):
    def __init__(self, config, load_plm_weights=False):
        super().__init__(config)
        # pretrained model path
        self.model_name_or_path = config.model_name_or_path
        # hyperparameters
        self.num_labels = config.num_labels
        self.num_embeddings = config.num_embeddings
        self.hidden_size = config.hidden_size
        # label attention / cross attention
        self.use_cross_attention = config.use_cross_attention
        self.bce_loss_fn = BCEWithLogitsLoss()
        # Synthetic Guidance
        self.use_guidance = config.use_guidance
        self.use_sim_loss = config.use_sim_loss
        self.lambda_sim_loss = config.lambda_sim_loss

        # Tricks
        # Biaffine
        self.use_biaffine = config.use_biaffine
        # R_drop
        self.use_rdrop = config.use_rdrop
        self.rdrop_alpha = config.rdrop_alpha

        if load_plm_weights:
            self.text_encoder = AutoModel.from_pretrained(
                self.model_name_or_path,
                config=config,
            )
        else:
            self.text_encoder = RobertaModel(config=config)

        self.retrieve_embedding = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=self.config.initializer_range,
                size=(self.num_embeddings, self.hidden_size),
            )
        )
        self.classify_embedding = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=self.config.initializer_range,
                size=(self.num_labels, self.hidden_size),
            )
        )

        if self.use_cross_attention:
            self.cross_attention = CrossAttention(self.hidden_size)
        else:
            self.cross_attention = LabelAttention(self.hidden_size)

        self.classifier = LinearClassifier(
            self.num_labels, self.hidden_size, use_biaffine=self.use_biaffine
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        num_chunks=None,
        label=None,
        soft_label=None,
        guidance_input_ids=None,
        guidance_attention_mask=None,
        guidance_num_chunks=None,
        **kwargs,
    ):
        """
        input_ids: (num_chunks, chunk_size)
        attention_mask: (num_chunks, chunk_size)
        num_chunks: list[int]
        return: (bs, num_chunks * chunk_size, hidden_size)
        """

        if self.training and self.use_guidance:
            input_ids = torch.cat([input_ids, guidance_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, guidance_attention_mask], dim=0)
            num_chunks += guidance_num_chunks
            label = torch.cat([label, label], dim=0)

        if self.training and self.use_rdrop:
            # use rdrop
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            num_chunks += num_chunks
            label = torch.cat([label, label], dim=0)

        _, chunk_size = input_ids.size()
        device = input_ids.device
        max_token_length = max(num_chunks) * chunk_size
        # print(input_ids)
        # exit()
        text_features = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        # print(text_features)
        # exit()

        chunk_idx = 0
        text_feature_list, attention_mask_list = [], []
        for num_chunk in num_chunks:
            text_feature_list.append(
                torch.cat(
                    [
                        text_features[chunk_idx : chunk_idx + num_chunk].view(
                            num_chunk * chunk_size, self.hidden_size
                        ),
                        torch.zeros(
                            max_token_length - num_chunk * chunk_size, self.hidden_size
                        ).to(device),
                    ]
                )
            )
            attention_mask_list.append(
                torch.cat(
                    [
                        attention_mask[chunk_idx : chunk_idx + num_chunk].view(
                            num_chunk * chunk_size
                        ),
                        torch.zeros(max_token_length - num_chunk * chunk_size).to(
                            device
                        ),
                    ]
                )
            )
            chunk_idx += num_chunk
        text_features = torch.stack(text_feature_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        retrieve_embedding = self.retrieve_embedding

        # (bs, num_codes, length), (bs, num_labels, hidden_size)
        att_weights, retrieved_features = self.cross_attention(
            retrieve_embedding, text_features, attention_mask
        )

        if self.use_biaffine:
            logits = self.classifier(retrieve_embedding, retrieved_features)
        else:
            logits = self.classifier(self.classify_embedding, retrieved_features)

        if label is not None:
            bce_loss = self.bce_loss_fn(logits, label)
            if self.training and self.use_rdrop:
                # (bs * 2, num_labels)
                logits_a, logits_b = logits.chunk(2)
                rdrop_loss = self.kl_loss(logits_a, logits_b)
            else:
                rdrop_loss = 0

            if self.training and self.use_guidance and self.use_sim_loss:
                # (2 * num_positive_labels, hidden_size)
                evidence = retrieved_features[label.bool()]
                # (num_positive_labels, hidden_size)
                if self.use_rdrop:
                    (
                        sample_evidence1,
                        guidance_evidence1,
                        sample_evidence2,
                        guidance_evidence2,
                    ) = evidence.chunk(4)
                    sim_loss = (
                        1
                        - F.cosine_similarity(
                            sample_evidence1, guidance_evidence1
                        ).mean()
                        + 1
                        - F.cosine_similarity(
                            sample_evidence2, guidance_evidence2
                        ).mean()
                    ) / 2
                else:
                    sample_evidence, guideline_evidence = evidence.chunk(2)
                    sim_loss = (
                        1
                        - F.cosine_similarity(
                            sample_evidence, guideline_evidence
                        ).mean()
                    )
            else:
                sim_loss = 0

            loss = (
                bce_loss
                + self.rdrop_alpha * rdrop_loss
                + self.lambda_sim_loss * sim_loss
            )
        else:
            bce_loss = None
            sim_loss = None
            loss = None

        return Output(
            logits=logits,
            loss=loss,
            bce_loss=bce_loss,
            sim_loss=sim_loss,
            rdrop_loss=rdrop_loss,
            attention_weights=att_weights,
        )

    def kl_loss(self, p, q):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
        )
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss

    @torch.no_grad()
    def retrieve_embed_init(self, idx, input_ids, attention_mask):
        device = self.text_encoder.device
        text_features = self.text_encoder(
            input_ids.to(device), attention_mask=attention_mask.to(device)
        )[0].max(dim=1)[0]
        self.retrieve_embedding[idx] = text_features

    @torch.no_grad()
    def classifier_embed_init(self):
        self.classify_embedding[:] = self.retrieve_embedding[: self.num_labels]

    @torch.no_grad()
    def get_topk_pred(self, pred, k=8):
        topk_pred, topk_idx = torch.topk(pred, k, dim=1)
        topk_pred = torch.sigmoid(topk_pred)
        return topk_pred, topk_idx

    @torch.no_grad()
    def get_max_att_weight_and_pos(self, attention_weights):
        # (bs, num_labels, length) -> (bs, num_labels)
        max_values, max_indices = attention_weights.max(dim=2)
        return max_values, max_indices

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, code_features, text_features, attention_mask):
        num_codes = code_features.size(0)
        # (bs, num_code, length)
        att_weights = code_features @ self.w_k(text_features).transpose(1, 2)
        # (bs, num_codes, length)
        att_weights = nn.functional.softmax(att_weights, dim=-1)
        # (bs, num_codes, hidden_size)
        weighted_feature = att_weights @ self.w_v(text_features)
        # weighted_feature = att_weights @ text_features
        weighted_feature = self.layer_norm(weighted_feature)
        return att_weights, weighted_feature


class LabelAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, code_features, text_features, attention_mask):
        """
        text_features: (bs, num_chunks * chunk_size, hidden_size)
        code_features: (num_labels, hidden_size) if not use bios kg
                    or (bs, num_labels, hidden_size) if use bios kg
        """
        # (bs, num_chunks * chunk_size, hidden_size)
        projected = torch.tanh(self.proj(text_features))
        # (bs, num_chunks * chunk_size, num_labels)
        att_weights = projected @ code_features.transpose(-1, -2)

        num_labels = att_weights.size(2)
        # # padding mask
        # att_weights = att_weights.masked_fill(
        #     attention_mask.unsqueeze(-1).repeat(1, 1, num_labels) == 0,
        #     float("-inf"),
        # )
        # (bs, num_labels, num_chunks * chunk_size)
        att_weights = nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        # (bs, num_labels, hidden_size)
        weighted_output = att_weights @ text_features
        return att_weights, weighted_output


class LinearClassifier(nn.Module):
    def __init__(self, num_labels, hidden_size, use_biaffine=False):
        super().__init__()
        self.use_biaffine = use_biaffine
        if use_biaffine:
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, code_embedding, code_specific_features):
        if self.use_biaffine:
            # (num_labels, hidden_size) @ (bs, num_labels, hidden_size) -> (bs, num_labels)
            code_embedding = self.linear(code_embedding)
        # (num_labels, hidden_size) @ (bs, num_labels, hidden_size) -> (bs, num_labels)
        logits = code_embedding.mul(code_specific_features).sum(dim=2).add(self.bias)
        return logits


if __name__ == "__main__":
    pass
