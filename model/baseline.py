import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from allennlp.nn.util import batched_index_select
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Optional


class BaseBert(nn.Module):

    def __init__(self,
                 pretrained_model_name,
                 idiom_mask_length=4,
                 use_generation=True, ):
        """
        Args:
            pretrained_model_name: the name of the pretrained model
            idiom_mask_length: length of idiom mask
            use_generation: whether to use pretrained cls head
        """

        super().__init__()
        self.pre_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
        self.sentence_model = self.pre_model.bert
        self.use_generation = use_generation
        if use_generation:
            self.gen_cls = self.pre_model.cls
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            self.gen_cls = BertOnlyMLMHead(config)
        self.idiom_mask_length = idiom_mask_length

    def new_params(self):
        if self.use_generation:
            return []
        else:
            return [self.gen_cls]

    def pretrained_params(self):
        if self.use_generation:
            return [self.pre_model]
        else:
            return [self.sentence_model]

    def embed_sentence(self, input_ids, attention_mask, candidate_mask, candidate_index=None):
        r"""
                input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
                attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
                candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        """

        # (batch_size, sequence_length, hidden_size)
        sentence_outputs = self.sentence_model(input_ids, attention_mask=attention_mask)[0]
        # (batch_size, idiom_mask_length*hidden_size)
        idiom_outputs = torch.masked_select(sentence_outputs, candidate_mask.unsqueeze(-1)).reshape(
            -1, self.idiom_mask_length * sentence_outputs.shape[-1])

        idiom_pred = self.gen_cls(idiom_outputs.reshape(-1, sentence_outputs.shape[-1])).unsqueeze(-1)
        candidate_index = candidate_index.transpose(-1, -2)
        candidate_index = candidate_index.reshape(-1, candidate_index.shape[-1])
        idiom_logits = batched_index_select(idiom_pred, candidate_index).squeeze(-1)
        idiom_logits = idiom_logits.reshape(-1, self.idiom_mask_length, idiom_logits.shape[-1]).transpose(-1, -2)

        return idiom_logits

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            candidate_mask: Optional[torch.Tensor],
            candidate_pattern: Optional[torch.Tensor],
            candidate_pattern_mask: Optional[torch.Tensor],
            labels: Optional[torch.Tensor] = None,
    ):
        r"""
        input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
        attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`

        candidate_pattern: torch.LongTensor of shape `(batch_size, candidate_num, idiom_pattern_length)`
        candidate_pattern_mask: torch.LongTensor of shape `(batch_size, candidate_num, idiom_pattern_length)`

        labels: torch.LongTensor of shape `(batch_size, )`
        """

        # replace the [UNK] token with [MASK] token
        input_ids = input_ids.masked_fill(input_ids == 100, 103)

        candidate_index = candidate_pattern[:, :, 1:1 + self.idiom_mask_length]
        # (batch_size, hidden_size)
        idiom_logits = self.embed_sentence(input_ids, attention_mask, candidate_mask, candidate_index)

        if labels is not None:
            loss = F.cross_entropy(idiom_logits, labels.unsqueeze(-1).repeat(1, self.idiom_mask_length))
            return loss
        else:
            pt = F.log_softmax(idiom_logits, dim=-2).sum(-1)
            return pt.argmax(dim=-1)
