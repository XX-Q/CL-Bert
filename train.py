from typing import *
import os
import argparse
import numpy as np
import random
import tensorboard
import torch
import pytorch_lightning as pl
from model import ChIDModel
from dataloader import ChIDDataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="classify")
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
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_type", type=str, default="IC")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=15)

    args = parser.parse_args()
    if args.model_type == "classify":
        assert args.task_type == "IC"
    elif args.model_type in ["contrastive", "dual"]:
        assert args.task_type == "IE"
    else:
        raise ValueError("model_type must be in ['classify', 'contrastive', 'dual']")

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    dataloader = ChIDDataLoader(data_path=args.data_dir,
                                chid_file="train_data_5w.json",
                                batch_size=args.batch_size,
                                tokenizer_name=args.pretrained_model_name,
                                task_type=args.task_type,
                                max_length=args.max_length,
                                idiom_mask_length=args.idiom_mask_length,
                                num_workers=args.num_workers)

    t_total = len(dataloader.train_dataset) // args.batch_size * args.epoch

    model = ChIDModel(model_type=args.model_type,
                      pretrained_model_name=args.pretrained_model_name,
                      idiom_mask_length=args.idiom_mask_length,
                      idiom_vocab_size=args.idiom_vocab_size,
                      learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay,
                      warm_up_proportion=args.warm_up_proportion,
                      t_total=t_total)

    log_path = os.path.join(args.output_dir, args.model_type)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = pl.loggers.TensorBoardLogger(args.output_dir, name=args.model_type)
    checkpoint = pl.callbacks.ModelCheckpoint(os.path.join(args.output_dir, args.model_type, "checkpoints"),
                                              filename="{epoch}-{val_acc:.4f}", every_n_epochs=1, save_top_k=-1)
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
                         logger=logger,
                         callbacks=[checkpoint, progress_bar],
                         precision=16,
                         enable_progress_bar=True,
                         max_epochs=args.epoch,
                         min_epochs=10,
                         )
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
