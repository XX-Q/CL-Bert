import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
import torch
from utils import *
import json
from transformers import AutoTokenizer


class ChIDDatasetIC(Dataset):
    def __init__(self, data_path, chid_file="train_data_5w.json", idiom_file="idioms_ChID.json",
                 idiom_tag="#idiom#", idiom_mask_length=4, tokenizer_name="hfl/chinese-roberta-wwm-ext",
                 max_len=300, is_train=True):
        cache_path = os.path.join(data_path, '{}_{}_{}_IC.pkl'.format(chid_file[:-5],idiom_mask_length,max_len))
        cached = os.path.isfile(cache_path)
        if cached:
            with open(cache_path,'rb') as f:
                self.data = pickle.load(f)
        else:
            chid_path = os.path.join(data_path, 'ChID', chid_file)
            idiom_path = os.path.join(data_path, idiom_file)

            with open(chid_path, "r", encoding="utf-8") as f:
                raw_data = []
                for line in f.readlines():
                    raw_data.append(json.loads(line))

            with open(idiom_path, "r", encoding="utf-8") as f:
                idioms_dict = json.load(f)

            idiom_vocab = {w: i for i, w in enumerate(idioms_dict.keys())}

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,use_fast=True)

            self.data = []
            processed_data = []

            for single_raw_dat in tqdm(raw_data,desc='processing data from {}'.format(chid_file)):
                processed_data.extend(data_process(single_raw_dat,
                                                   idiom_tag=idiom_tag,
                                                   tokenizer=tokenizer,
                                                   length=max_len - 2,  # space for [CLS] and [EOS]
                                                   idiom_mask_length=idiom_mask_length,
                                                   replace_idiom=is_train))

            processed_sentence = []
            processed_idiom_ground_truth = []
            processed_idiom_candidate = []
            for single_processed_data in processed_data:
                processed_sentence.append(single_processed_data["sentence"])
                processed_idiom_ground_truth.append(single_processed_data["idiom_ground_truth"])
                processed_idiom_candidate.append([idiom_vocab[x] for x in single_processed_data["idiom_candidate"]])

            sentence_tokenization_output = tokenizer(processed_sentence,
                                                     padding="max_length",
                                                     truncation=True,
                                                     max_length=max_len,
                                                     return_tensors="pt")

            self.data = {
                "sentence_token": sentence_tokenization_output["input_ids"],
                # 1 for real token, 0 for padding
                "sentence_mask": sentence_tokenization_output["attention_mask"].bool(),
                # 1 for idiom mask tokens, 0 for others
                "idiom_token": (sentence_tokenization_output["input_ids"] == tokenizer.mask_token_id),
                "idiom_candidate_index": processed_idiom_candidate,
                # index in candidates
                "label": torch.tensor(processed_idiom_ground_truth)
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)

    def __getitem__(self, index):
        return self.data["sentence_token"][index], \
               self.data["sentence_mask"][index], \
               self.data["idiom_token"][index], \
               self.data["idiom_candidate_index"][index], \
               self.data["label"][index]

    def __len__(self):
        return len(self.data["sentence_token"])


if __name__ == "__main__":
    batch_size = 32
    if_shuffle = False

    myData = ChIDDatasetIC("../data/", chid_file="ChID/dev_data.json", is_train=True)
    myDataLoader = DataLoader(myData, batch_size=batch_size, shuffle=if_shuffle)

    print(len(myData), *myData[2])
