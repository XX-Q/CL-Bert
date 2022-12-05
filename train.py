from typing import *
import argparse
import numpy as np
import random
import torch
import pytorch_lightning as pl

from model import ClassifyBert, ContrastiveBert, DualBert
from dataset import ChIDDatasetIE, ChIDDatasetIC
from BertAdam import BertAdam


class ChIDModel(pl.LightningModule):
    def __init__(self,
                 model_type: Union["classify", "contrastive", "dual"] = "dual",
                 pretrained_model_name: str = "hfl/chinese-roberta-wwm-ext",
                 idiom_mask_length: int = 4,
                 idiom_vocab_size: int = 3848,
                 learning_rate: float = 1.5e-5,
                 weight_decay: float = 0.2,
                 warm_up_proportion: float = 0.05, ):
        if model_type == "classify":
            self.model = ClassifyBert(pretrained_model_name, idiom_mask_length, idiom_vocab_size)
        elif model_type == "dual":
            self.model = DualBert(pretrained_model_name, use_same_model=False, idiom_mask_length=idiom_mask_length)
        elif model_type == "contrastive":
            self.model = ContrastiveBert(pretrained_model_name, use_same_model=False,
                                         idiom_mask_length=idiom_mask_length)
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._warm_up_proportion = warm_up_proportion

    def configure_optimizers(self):
        return BertAdam(self.parameters(),
                        warmup=self._warm_up_proportion,
                        lr=self._lr,
                        weight_decay=self._weight_decay,
                        t_total=len(self.train_dataloader()) * self.trainer.max_epochs)

    def training_step(self, batch, batch_idx):
        loss = self.model(*batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        label = batch[-1]
        predict = self.model(*batch[:-1])
        return {"predict": predict.cpu(), "label": label.cpu()}

    def validation_epoch_end(self, outputs):
        predict = torch.cat([output["predict"] for output in outputs], dim=0)
        label = torch.cat([output["label"] for output in outputs], dim=0)
        acc = (predict.argmax(dim=-1) == label).float().mean()
        self.log("val_acc", acc, prog_bar=True, logger=True, on_epoch=True)
        return {"val_acc": acc}

    def test_step(self, batch, batch_idx):
        label = batch[-1]
        predict = self.model(*batch[:-1])
        return {"predict": predict.cpu(), "label": label.cpu()}

    def test_epoch_end(self, outputs):
        predict = torch.cat([output["predict"] for output in outputs], dim=0)
        label = torch.cat([output["label"] for output in outputs], dim=0)
        acc = (predict.argmax(dim=-1) == label).float().mean()
        self.log("test_acc", acc, prog_bar=True, logger=True, on_epoch=True)
        return {"test_acc": acc}

    def forward(self, datas):
        return self.model(*datas)


class ChIDDataset(pl.LightningDataModule):
    def __init__(self, data_path, chid_file, batch_size, tokenizer_name, task_type, max_length=300, idiom_mask_length=4,
                 num_workers=8):
        super().__init__()
        self._data_path = data_path
        self._chid_file = chid_file
        self._batch_size = batch_size
        self._tokenizer_name = tokenizer_name
        self._max_length = max_length
        self._num_workers = num_workers
        self._idiom_mask_length = idiom_mask_length
        self._task_type = task_type

    def train_dataloader(self):
        if self._task_type == "IE":
            train_dataset = ChIDDatasetIE(self._data_path,
                                          chid_file=self._chid_file,
                                          tokenizer_name=self._tokenizer_name,
                                          max_len=self._max_length,
                                          idiom_mask_length=self._idiom_mask_length,
                                          is_train=True)
        elif self._task_type == "IC":
            train_dataset = ChIDDatasetIC(self._data_path,
                                          chid_file=self._chid_file,
                                          tokenizer_name=self._tokenizer_name,
                                          max_len=self._max_length,
                                          idiom_mask_length=self._idiom_mask_length,
                                          is_train=True)
        else:
            raise ValueError("task_type must be IE or IC")

        return torch.utils.data.DataLoader(train_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=True,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           drop_last=True)

    def val_dataloader(self):
        if self._task_type == "IE":
            val_dataset = ChIDDatasetIE(self._data_path,
                                        chid_file="dev_data.json",
                                        tokenizer_name=self._tokenizer_name,
                                        max_len=self._max_length,
                                        idiom_mask_length=self._idiom_mask_length,
                                        is_train=False)
        elif self._task_type == "IC":
            val_dataset = ChIDDatasetIC(self._data_path,
                                        chid_file="dev_data.json",
                                        tokenizer_name=self._tokenizer_name,
                                        max_len=self._max_length,
                                        idiom_mask_length=self._idiom_mask_length,
                                        is_train=False)
        else:
            raise ValueError("task_type must be IE or IC")

        return torch.utils.data.DataLoader(val_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           drop_last=False)

    def test_dataloader(self):
        if self._task_type == "IE":
            test_dataset = ChIDDatasetIE(self._data_path,
                                         chid_file="test_data.json",
                                         tokenizer_name=self._tokenizer_name,
                                         max_len=self._max_length,
                                         idiom_mask_length=self._idiom_mask_length,
                                         is_train=False)
        elif self._task_type == "IC":
            test_dataset = ChIDDatasetIC(self._data_path,
                                         chid_file="test_data.json",
                                         tokenizer_name=self._tokenizer_name,
                                         max_len=self._max_length,
                                         idiom_mask_length=self._idiom_mask_length,
                                         is_train=False)
        else:
            raise ValueError("task_type must be IE or IC")

        return torch.utils.data.DataLoader(test_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           drop_last=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="dual")
    parser.add_argument("--pretrained_model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--idiom_mask_length", type=int, default=4)
    parser.add_argument("--idiom_vocab_size", type=int, default=3848)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--warm_up_proportion", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_type", type=str, default="IE")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    model = ChIDModel(model_type=args.model_type,
                      pretrained_model_name=args.pretrained_model_name,
                      idiom_mask_length=args.idiom_mask_length,
                      idiom_vocab_size=args.idiom_vocab_size,
                      learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay,
                      warm_up_proportion=args.warm_up_proportion)

    dataloader = ChIDDataset(data_path=args.data_dir,
                             chid_file="train_data_5w.json",
                             batch_size=args.batch_size,
                             tokenizer_name=args.pretrained_model_name,
                             task_type=args.task_type,
                             max_length=args.max_length,
                             idiom_mask_length=args.idiom_mask_length,
                             num_workers=args.num_workers)

    logger = pl.loggers.TensorBoardLogger("logs", name="ChID")
    checkpoint = pl.callbacks.ModelCheckpoint("checkpoints", filename="{epoch}-{val_acc:.4f}", every_n_epochs=1)
    progress_bar = pl.callbacks.RichProgressBar(
        theme=pl.callbacks.progress.rich_progress.RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    trainer = pl.Trainer(accelerator='gpu',
                         strategy='ddp',
                         gpus=-1,
                         logger=logger,
                         callbacks=[checkpoint, progress_bar],
                         auto_scale_batch_size='binsearch',
                         precision=16,
                         enable_progress_bar=True,
                         max_epochs=args.epochs,
                         min_epochs=10,
                         )
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
