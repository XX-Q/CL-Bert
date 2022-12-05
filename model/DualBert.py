import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from typing import Optional


class DualBert(nn.Module):

    def __init__(self,
                 pretrained_model_name,
                 use_same_model=False,
                 idiom_mask_length=4,
                 mode='cosine_similarity',
                 linear_hidden_size=256):

        super(DualBert, self).__init__()

        self.sentence_model = BertModel.from_pretrained(pretrained_model_name)
        if use_same_model:
            self.idiom_model = self.sentence_model
        else:
            self.idiom_model = BertModel.from_pretrained(pretrained_model_name)

        self.sentence_pooler = nn.Sequential(
            nn.Linear(self.sentence_model.config.hidden_size * idiom_mask_length,
                      self.sentence_model.config.hidden_size),
            nn.Tanh()
        )
        self.mode = mode
        if self.mode == 'linear':
            self.aggregation = nn.Sequential(
                nn.Linear(self.sentence_model.config.hidden_size + self.idiom_model.hidden_size,
                          linear_hidden_size),
                nn.Tanh(),
                nn.Linear(linear_hidden_size, 1)
            )

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
        idiom_outputs = idiom_outputs / torch.norm(idiom_outputs, dim=-1, keepdim=True)
        return idiom_outputs

    def embed_idiom_pattern(self, input_ids, attention_mask):
        r"""
                input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
                attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        """
        # (batch_size, hidden_size)
        idiom_pattern_outputs = self.idiom_model(input_ids, attention_mask=attention_mask)[1]
        idiom_pattern_outputs = idiom_pattern_outputs / torch.norm(idiom_pattern_outputs, dim=1, keepdim=True)

        return idiom_pattern_outputs

    def calculate_logits(self, sentence_outputs, idiom_pattern_outputs):
        r"""
                sentence_outputs: torch.FloatTensor of shape `(batch_size, hidden_size)`
                idiom_pattern_outputs: torch.FloatTensor of shape `(batch_size, candidate_num, hidden_size)`
        """
        if self.mode == 'cosine_similarity':
            logits = torch.matmul(sentence_outputs.unsqueeze(-2), idiom_pattern_outputs.transpose(1, 2)).squeeze(-2)
        elif self.mode == 'euclidean_distance':
            logits = -torch.norm(sentence_outputs.unsqueeze(-2) - idiom_pattern_outputs, dim=-1)
        elif self.mode == 'linear':
            logits = self.aggregation(torch.cat([sentence_outputs.unsqueeze(-2), idiom_pattern_outputs],
                                                dim=-1)).squeeze(-1)
        else:
            raise ValueError('mode must be cosine_similarity or euclidean_distance or linear')
        return logits

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            candidate_mask: Optional[torch.Tensor],
            candidate_pattern: Optional[torch.Tensor],
            candidate_pattern_mask: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
    ):
        r"""
        input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
        attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`

        candidate_pattern: torch.LongTensor of shape `(batch_size, candidate_num, idiom_pattern_length)`
        candidate_pattern_mask: torch.LongTensor of shape `(batch_size, candidate_num, idiom_pattern_length)`

        labels: torch.LongTensor of shape `(batch_size, )`
        """

        # (batch_size, hidden_size)
        idiom_outputs = self.embed_sentence(input_ids, attention_mask, candidate_mask)

        B, N, L = candidate_pattern.shape
        candidate_pattern = candidate_pattern.reshape(-1, L)
        candidate_pattern_mask = candidate_pattern_mask.reshape(-1, L)
        # (batch_size*candidate_num, hidden_size)
        idiom_pattern_outputs = self.embed_idiom_pattern(candidate_pattern, candidate_pattern_mask)

        # (batch_size, candidate_num, hidden_size)
        candidate_pattern_outputs = idiom_pattern_outputs.reshape(B, N, -1)
        # (batch_size, candidate_num)
        logits = self.calculate_logits(idiom_outputs, candidate_pattern_outputs)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss
        else:
            return logits.argmax(dim=-1)
