from typing import *
import torch
import pytorch_lightning as pl
from torch.optim import AdamW, lr_scheduler

from .ClassifyBert import ClassifyBert
from .DualBert import DualBert
from .ContrastiveBert import ContrastiveBert


class ChIDModel(pl.LightningModule):
    def __init__(self,
                 model_type: Union[Literal["classify"], Literal["contrastive"], Literal["dual"]] = "dual",
                 pretrained_model_name: str = "hfl/chinese-roberta-wwm-ext",
                 idiom_mask_length: int = 4,
                 idiom_vocab_size: int = 3848,
                 idiom_use_cls: bool = True,
                 learning_rate: float = 1.5e-5,
                 weight_decay: float = 0.2,
                 warm_up_proportion: float = 0.05,
                 t_total: int = 100000, ):
        super().__init__()
        self.optim = None
        if model_type == "classify":
            self.core_model = ClassifyBert(pretrained_model_name, idiom_mask_length, idiom_vocab_size)
        elif model_type == "dual":
            self.core_model = DualBert(pretrained_model_name, use_same_model=False,
                                       idiom_mask_length=idiom_mask_length, use_cls=idiom_use_cls)
        elif model_type == "contrastive":
            self.core_model = ContrastiveBert(pretrained_model_name, use_same_model=False,
                                              idiom_mask_length=idiom_mask_length, use_cls=idiom_use_cls)
        self._lr = learning_rate
        self.new_lr = learning_rate * 5
        self._weight_decay = weight_decay
        self._warm_up_proportion = warm_up_proportion
        self._t_total = t_total

    def configure_optimizers(self):
        params_group = []
        for p in self.core_model.new_params():
            if isinstance(p, torch.nn.Parameter):
                params = {'params': p}
            else:
                params = {'params': p.parameters()}
            params['lr'] = self.new_lr
            params_group.append(params)
        for p in self.core_model.pretrained_params():
            params = {'params': p.parameters(), 'lr': self._lr}
            params_group.append(params)
        optim = AdamW(params_group, weight_decay=self._weight_decay)
        warm_up_steps = int(self._warm_up_proportion * self._t_total)
        sched = lr_scheduler.CosineAnnealingWarmRestarts(optim, warm_up_steps, self._t_total)

        return {"optimizer": optim, "lr_scheduler": sched}

    def training_step(self, batch, batch_idx):
        loss = self.core_model(*batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        label = batch[-1]
        predict = self.core_model(*batch[:-1])
        return {"predict": predict.cpu(), "label": label.cpu()}

    def validation_epoch_end(self, outputs):
        predict = torch.cat([output["predict"] for output in outputs], dim=0)
        label = torch.cat([output["label"] for output in outputs], dim=0)
        acc = (predict == label).float().mean()
        self.log("val_acc", acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        print('model saved!')
        return {"val_acc": acc}

    def test_step(self, batch, batch_idx):
        label = batch[-1]
        predict = self.core_model(*batch[:-1])
        return {"predict": predict.cpu(), "label": label.cpu()}

    def test_epoch_end(self, outputs):
        predict = torch.cat([output["predict"] for output in outputs], dim=0)
        label = torch.cat([output["label"] for output in outputs], dim=0)
        acc = (predict == label).float().mean()
        self.log("test_acc", acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return {"test_acc": acc}

    def forward(self, datas):
        return self.core_model(*datas)