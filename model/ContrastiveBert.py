import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from allennlp.nn.util import batched_index_select
from transformers import BertModel, AutoModelForMaskedLM
from typing import Optional


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        local_rank = torch.distributed.get_rank()

        output = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(output, tensor)

        ctx.rank = local_rank
        ctx.batch_size = tensor.shape[0]

        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
        )


allgather = AllGather.apply


class ContrastiveBert(nn.Module):

    def __init__(self,
                 pretrained_model_name,
                 use_same_model=False,
                 idiom_mask_length=4,
                 use_cls=True,
                 use_mask=False,
                 use_generation=True):
        """
        Args:
            pretrained_model_name: name of pretrained model
            use_same_model: whether to use the same model for sentence and idiom pattern
            idiom_mask_length: length of idiom mask
            use_cls: whether to use the first token of idiom pattern
            use_generation: whether to use generation model to predict idiom
        """

        super(ContrastiveBert, self).__init__()

        self.bert_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
        self.sentence_model = self.bert_model.bert
        if use_same_model:
            self.idiom_model = self.sentence_model
        else:
            self.idiom_model = BertModel.from_pretrained(pretrained_model_name)

        self.sentence_pooler = nn.Sequential(
            nn.Linear(self.sentence_model.config.hidden_size * idiom_mask_length,
                      self.sentence_model.config.hidden_size),
            nn.Tanh()
        )

        self.use_cls = use_cls
        if not use_cls:
            self.idiom_pooler = nn.Sequential(
                nn.Linear(self.idiom_model.config.hidden_size * idiom_mask_length,
                          self.idiom_model.config.hidden_size),
                nn.Tanh()
            )

        self.use_mask = use_mask
        self.use_generation = use_generation
        if use_generation:
            self.gen_cls = self.bert_model.cls

        self.idiom_mask_length = idiom_mask_length
        self.use_same_model = use_same_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def new_params(self):
        return [self.sentence_pooler, self.logit_scale] if self.use_cls \
            else [self.sentence_pooler, self.idiom_pooler, self.logit_scale]

    def pretrained_params(self):
        if self.use_generation:
            params = [self.bert_model]
        else:
            params = [self.sentence_model]

        if self.use_same_model:
            params.append(self.idiom_model)

        return params

    def embed_sentence(self, input_ids, attention_mask, candidate_mask, candidate_index=None):
        """
        Args:
            input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
            attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
            candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
            candidate_index: torch.LongTensor of shape `(batch_size, candidate_num, idiom_mask_length)`
        """

        # (batch_size, sequence_length, hidden_size)
        sentence_outputs = self.sentence_model(input_ids, attention_mask=attention_mask)[0]
        # (batch_size, idiom_mask_length*hidden_size)
        idiom_outputs = torch.masked_select(sentence_outputs, candidate_mask.unsqueeze(-1)).reshape(
            -1, self.idiom_mask_length * sentence_outputs.shape[-1])

        if self.use_generation:
            # (batch_size * idiom_mask_length, vocab_size)
            idiom_pred = self.gen_cls(idiom_outputs.reshape(-1, sentence_outputs.shape[-1])).unsqueeze(-1)

            # (batch_size * idiom_mask_length, candidate_num)
            candidate_index = candidate_index.transpose(-1, -2)
            candidate_index = candidate_index.reshape(-1, candidate_index.shape[-1])

            # (batch_size * idiom_mask_length, candidate_num)
            idiom_logits = batched_index_select(idiom_pred, candidate_index).squeeze(-1)
            # (batch_size, candidate_num, idiom_mask_length)
            idiom_logits = idiom_logits.reshape(-1, self.idiom_mask_length, idiom_logits.shape[-1]).transpose(-1, -2)
        else:
            idiom_logits = None

        # (batch_size, hidden_size)
        idiom_outputs = self.sentence_pooler(idiom_outputs)
        idiom_outputs = idiom_outputs / torch.norm(idiom_outputs, dim=-1, keepdim=True)
        if self.use_generation:
            return idiom_outputs, idiom_logits
        else:
            return idiom_outputs

    def embed_idiom_pattern(self, input_ids, attention_mask):
        """
        Args:
            input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
            attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        """
        # (batch_size, hidden_size)
        if self.use_cls:
            idiom_pattern_outputs = self.idiom_model(input_ids, attention_mask=attention_mask)[1]
            idiom_pattern_outputs = idiom_pattern_outputs / torch.norm(idiom_pattern_outputs, dim=1, keepdim=True)
        else:
            # (batch_size, sequence_length, hidden_size)
            idiom_pattern_outputs = self.idiom_model(input_ids, attention_mask=attention_mask)[0]
            # (batch_size, idiom_mask_length*hidden_size)
            idiom_pattern_outputs = idiom_pattern_outputs[:, 1:1 + self.idiom_mask_length, :].reshape(
                -1, self.idiom_mask_length * idiom_pattern_outputs.shape[-1])
            # (batch_size, hidden_size)
            idiom_pattern_outputs = self.idiom_pooler(idiom_pattern_outputs)
            idiom_pattern_outputs = idiom_pattern_outputs / torch.norm(idiom_pattern_outputs, dim=1, keepdim=True)

        return idiom_pattern_outputs

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
        with torch.no_grad():
            self.logit_scale.data.clamp_(-100, 100)
        if self.use_generation:
            candidate_index = candidate_pattern[:, :, 1:1 + self.idiom_mask_length]
            # (batch_size, hidden_size), (batch_size, candidate_num, idiom_mask_length)
            idiom_outputs, idiom_logits = self.embed_sentence(input_ids, attention_mask, candidate_mask,
                                                              candidate_index)
        else:
            # (batch_size, hidden_size)
            idiom_outputs = self.embed_sentence(input_ids, attention_mask, candidate_mask)
            idiom_logits = None

        if self.use_mask:
            candidate_pattern[:, :, 1:1 + self.idiom_mask_length] = 103  # [MASK] token id
        B, N, L = candidate_pattern.shape
        candidate_pattern = candidate_pattern.reshape(-1, L)
        candidate_pattern_mask = candidate_pattern_mask.reshape(-1, L)
        # (batch_size*candidate_num, hidden_size)
        idiom_pattern_outputs = self.embed_idiom_pattern(candidate_pattern, candidate_pattern_mask)

        if labels is not None:

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                labels = allgather(labels)
                idiom_pattern_outputs = allgather(idiom_pattern_outputs)
                idiom_outputs = allgather(idiom_outputs)

            # (batch_size, candidate_num*batch_size)
            i2c_matrix = torch.mm(idiom_outputs, idiom_pattern_outputs.t())
            i2c_matrix *= self.logit_scale
            c2i_matrix = i2c_matrix.t()

            # (batch_size, )
            idx = torch.arange(B, device=idiom_outputs.device)
            labels = labels + idx * N

            # (batch_size, candidate_num*batch_size)
            i2c_pt = i2c_matrix.softmax(dim=-1) + 1e-8
            # (batch_size, )
            i2c_pt = i2c_pt[idx, labels]
            i2c_logpt = i2c_pt.log()
            i2c_loss = -i2c_logpt.mean()

            # (candidate_num*batch_size, batch_size)
            c2i_pt = c2i_matrix.softmax(dim=-1) + 1e-8
            # (batch_size, )
            c2i_pt = c2i_pt[labels, idx]
            c2i_logpt = c2i_pt.log()
            c2i_loss = -c2i_logpt.mean()

            loss = (i2c_loss + c2i_loss) / 2

            if self.use_generation:
                loss += F.cross_entropy(idiom_logits, labels.unsqueeze(-1).repeat(1, self.idiom_mask_length))
            return loss

        else:
            idiom_pattern_outputs = idiom_pattern_outputs.reshape(B, N, -1)
            # (batch_size, candidate_num)
            i2c_matrix = torch.matmul(idiom_outputs.unsqueeze(-2), idiom_pattern_outputs.transpose(-1, -2)).squeeze(-2)
            i2c_matrix *= self.logit_scale
            pt = i2c_matrix.softmax(dim=-1)
            if self.use_generation:
                idiom_pt = F.softmax(idiom_logits, dim=-2)
                pt1 = pt + idiom_pt.sum(-1)
                pt2 = pt + idiom_pt.mean(-1)
                pt3 = self.idiom_mask_length * pt.log() + idiom_pt.log().sum(-1)
                return torch.stack((pt1.argmax(-1), pt2.argmax(-1), pt3.argmax(-1)), dim=-1)
            else:
                return pt.argmax(-1)
