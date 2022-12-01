import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn.util import batched_index_select
from transformers import BertModel
from typing import Optional


class ClassifyBert(nn.Module):

    def __init__(self, pretrained_model_name, idiom_mask_length, idiom_vocab_size):

        super(ClassifyBert, self).__init__()

        self.sentence_model = BertModel.from_pretrained(pretrained_model_name)
        self.sentence_pooler = nn.Sequential(
            nn.Linear(self.sentence_model.config.hidden_size * idiom_mask_length,
                      self.sentence_model.config.hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(self.sentence_model.config.hidden_size, idiom_vocab_size)

        self.idiom_mask_length = idiom_mask_length

    def embed_sentence(self, input_ids, attention_mask, candidate_mask):
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
        # (batch_size, hidden_size)
        idiom_outputs = self.sentence_pooler(idiom_outputs)
        idiom_logits = self.cls(idiom_outputs)

        return idiom_outputs

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            candidate_mask: Optional[torch.Tensor],
            candidate_index: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
    ):
        r"""
        input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
        attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`

        candidate_index: torch.LongTensor of shape `(batch_size, candidate_num)`

        labels: torch.LongTensor of shape `(batch_size, )`
        """

        # (batch_size, idiom_vocab_size, 1)
        idiom_logits = self.embed_sentence(input_ids, attention_mask, candidate_mask).unsqueeze(-1)

        # (batch_size, candidate_num)
        idiom_logits = batched_index_select(idiom_logits, candidate_index).squeeze(-1)
        if labels is not None:
            loss = F.cross_entropy(idiom_logits, labels)
            return loss
        else:
            return idiom_logits.argmax(-1)