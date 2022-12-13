from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import torch
import pickle
from tqdm import tqdm
import os
from utils import *
import json
from transformers import AutoTokenizer


class ChIDDatasetIE(Dataset):
    def __init__(self,
                 data_path,
                 chid_file="train_data_5w.json",
                 idiom_file="idioms_ChID.json",
                 idiom_tag="#idiom#",
                 idiom_mask_length=4,
                 tokenizer_name="hfl/chinese-roberta-wwm-ext",
                 max_len=300,
                 idiom_pattern_max_length=64,
                 replace_idiom=False):
        """
        Args:
            data_path: the path of the data
            chid_file: the name of the ChID file
            idiom_file: the name of the idiom dictionary file
            idiom_tag: the tag of the idiom in the ChID file
            idiom_mask_length: the length of the idiom mask
            tokenizer_name: the name of the tokenizer
            max_len: the max length of the sentence
            idiom_pattern_max_length: the max length of the idiom pattern
            replace_idiom: whether to replace the idiom not predicted by the model with the ground truth
        """

        dataset_name = '{}_{}_{}_{}_{}_{}_IE'.format(chid_file[:-5],
                                                     idiom_mask_length,
                                                     tokenizer_name.replace('/', '_'),
                                                     max_len,
                                                     idiom_pattern_max_length,
                                                     'R' if replace_idiom else 'NR')

        cache_path = os.path.join(data_path, dataset_name + '.pkl')
        cached = os.path.isfile(cache_path)
        if cached:
            print("Loading cached data from {}".format(cache_path))
            with open(cache_path, 'rb') as f:
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

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

            self.data = []
            processed_data = []

            for single_raw_dat in tqdm(raw_data, desc='processing data from {}'.format(chid_file)):
                processed_data.extend(data_process(single_raw_dat,
                                                   idiom_tag=idiom_tag,
                                                   tokenizer=tokenizer,
                                                   length=max_len - 2,  # space for [CLS] and [EOS]
                                                   idiom_mask_length=idiom_mask_length,
                                                   replace_idiom=replace_idiom))

            processed_sentence = []
            processed_idiom_ground_truth = []
            processed_idiom_candidate_pattern = []
            for single_processed_data in processed_data:
                processed_sentence.append(single_processed_data["sentence"])
                processed_idiom_ground_truth.append(single_processed_data["idiom_ground_truth"])
                processed_idiom_candidate_pattern.extend([x + "：" + idioms_dict[x]
                                                          for x in single_processed_data["idiom_candidate"]])

            sentence_tokenization_output = tokenizer(processed_sentence,
                                                     padding="max_length",
                                                     truncation=True,
                                                     max_length=max_len,
                                                     return_tensors="pt")

            idiom_tokenization_output = tokenizer(processed_idiom_candidate_pattern,
                                                  padding="max_length",
                                                  truncation=True,
                                                  max_length=idiom_pattern_max_length,
                                                  return_tensors="pt")

            self.data = {
                "sentence_token":
                    sentence_tokenization_output["input_ids"],
                # True for real token, False for padding token
                "sentence_mask":
                    sentence_tokenization_output["attention_mask"].bool(),
                # True for idiom, False for non-idiom
                "idiom_token":
                    (sentence_tokenization_output["input_ids"] == tokenizer.mask_token_id),
                "idiom_candidate_pattern_token":
                    idiom_tokenization_output["input_ids"]
                    .reshape(-1, len(processed_idiom_candidate_pattern) // len(processed_sentence),
                             idiom_pattern_max_length),
                # True for real token, False for padding
                "idiom_candidate_pattern_mask":
                    idiom_tokenization_output["attention_mask"]
                    .reshape(-1, len(processed_idiom_candidate_pattern) // len(processed_sentence),
                             idiom_pattern_max_length).bool(),
                # index in candidates
                "label": torch.tensor(processed_idiom_ground_truth)
            }
            print("Saving cached data to {}".format(cache_path))
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)

    def __getitem__(self, index):
        return self.data["sentence_token"][index], \
               self.data["sentence_mask"][index], \
               self.data["idiom_token"][index], \
               self.data["idiom_candidate_pattern_token"][index], \
               self.data["idiom_candidate_pattern_mask"][index], \
               self.data["label"][index]

    def __len__(self):
        return len(self.data["sentence_token"])


if __name__ == "__main__":
    batch_size = 32
    if_shuffle = False

    myData = ChIDDatasetIE("../data/", chid_file="ChID/dev_data.json", replace_idiom=True)
    myDataLoader = DataLoader(myData, batch_size=batch_size, shuffle=if_shuffle)

    print(len(myData), *myData[2])
