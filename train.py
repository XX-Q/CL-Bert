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
    parser.add_argument("--model_type", type=str, default="dual",
                        help="model type", choices=["dual", "classify", "contrastive", "baseline"])
    parser.add_argument("--task_type", type=str, default="IE",
                        help="task type Idiom Classification or Idiom Explanation", choices=["IC", "IE"])

    parser.add_argument("--data_dir", type=str, default="./data",
                        help="data directory")
    parser.add_argument("--chid_file", type=str, default="train_data_5w.json",
                        help="chid file for training")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp",
                        help="output directory for logging and saving checkpoints")

    parser.add_argument("--max_length", type=int, default=512,
                        help="max length of input sequence")
    parser.add_argument("--idiom_mask_length", type=int, default=4,
                        help="idiom mask length for idiom masking")
    parser.add_argument("--replace_idiom", action="store_true", default=False,
                        help="replace idiom with ground truth or not")

    parser.add_argument("--epoch", type=int, default=10,
                        help="epoch num for training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training")
    parser.add_argument("--gpus", type=int, default=1,
                        help="number of gpus to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    parser.add_argument("--precision", type=int, default=16,
                        help="precision for training")

    parser.add_argument("--pretrained_model_name", type=str, default="hfl/chinese-roberta-wwm-ext",
                        help="pretrained model name")
    parser.add_argument("--idiom_vocab_size", type=int, default=3848,
                        help="idiom vocabulary size")
    parser.add_argument("--idiom_use_cls", action="store_true", default=False,
                        help="use cls token for idiom pattern embedding or use idiom tokens")
    parser.add_argument("--idiom_use_mask", action="store_true", default=False,
                        help="use mask token for idiom pattern embedding or use idiom tokens")

    parser.add_argument("--sim_mode", type=str, default="cosine_similarity",
                        help="similarity mode for idiom pattern similarity",
                        choices=['cosine_similarity', 'euclidean_distance', 'linear', 'cross_attention'])
    parser.add_argument("--linear_hidden_size", type=int, default=256,
                        help="hidden size for linear layer in linear mode dual model")
    parser.add_argument("--use_pretrained_generation", action="store_true", default=False,
                        help="use pretrained generation cls head or not")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--learning_rate", type=float, default=1.5e-5,
                        help="learning rate for new-added layer training")
    parser.add_argument("--fine_tune_learning_rate", type=float, default=1.5e-5,
                        help="learning rate for pretrained model fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.2,
                        help="weight decay for optimizer")
    parser.add_argument("--warm_up_proportion", type=float, default=0.05,
                        help="warm up proportion for optimizer")

    args = parser.parse_args()
    if args.model_type == "classify":
        assert args.task_type == "IC"
    elif args.model_type in ["contrastive", "dual", "baseline"]:
        assert args.task_type == "IE"
    else:
        raise ValueError("model_type must be in ['classify', 'contrastive', 'dual', 'baseline']")

    args_str = "args:\n"
    for k, v in args.__dict__.items():
        args_str += f"{k}: {v}\t"
    print(args_str)

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)  # "train_data_5w.json"
    dataloader = ChIDDataLoader(data_path=args.data_dir,
                                chid_file=args.chid_file,
                                batch_size=args.batch_size,
                                tokenizer_name=args.pretrained_model_name,
                                task_type=args.task_type,
                                max_length=args.max_length,
                                idiom_mask_length=args.idiom_mask_length,
                                replace_idiom=args.replace_idiom,
                                num_workers=args.num_workers)

    t_total = len(dataloader.train_dataset) // args.batch_size * args.epoch

    model = ChIDModel(model_type=args.model_type,
                      pretrained_model_name=args.pretrained_model_name,
                      idiom_mask_length=args.idiom_mask_length,
                      idiom_vocab_size=args.idiom_vocab_size,
                      idiom_use_cls=args.idiom_use_cls,
                      sim_mode=args.sim_mode,
                      linear_hidden_size=args.linear_hidden_size,
                      use_pretrained_generation=args.use_pretrained_generation,
                      learning_rate=args.learning_rate,
                      fine_tune_learning_rate=args.fine_tune_learning_rate,
                      weight_decay=args.weight_decay,
                      warm_up_proportion=args.warm_up_proportion,
                      t_total=t_total)

    log_path = os.path.join(args.output_dir, args.model_type)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = pl.loggers.TensorBoardLogger(args.output_dir, name=args.model_type)
    checkpoint = pl.callbacks.ModelCheckpoint(os.path.join(args.output_dir, args.model_type, "checkpoints"),
                                              filename="{epoch}-{val_acc:.4f}",
                                              every_n_epochs=1,
                                              save_top_k=1,
                                              monitor="val_acc",
                                              mode="max")
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
                         precision=args.precision,
                         enable_progress_bar=True,
                         max_epochs=args.epoch,
                         )
    trainer.fit(model, dataloader)
    # test the model on the best checkpoint
    trainer.test(ckpt_path="best", dataloaders=dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
