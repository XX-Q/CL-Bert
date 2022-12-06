from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import torch
from utils import *
import json
from transformers import AutoTokenizer


class IdiomsClassificationDataset(Dataset):
    def __init__(self, path, idioms_path="../data/new_idioms.json", tokenizer="hfl/chinese-roberta-wwm-ext"):
        with open(path, "r", encoding="utf-8") as f:
            raw_data = []
            for line in f.readlines():
                raw_data.append(json.loads(line))

        with open(idioms_path, "r") as f:
            idioms_dict = eval(f.read())

        with open("../data/idioms_list.json", "r", encoding="utf-8") as f:
            idioms_list = eval(f.read())

        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.data = []

        for single_raw_data in raw_data:
            # replace #idiom# with real idioms
            complete_content = single_raw_data["content"]
            for single_idiom_index in range(single_raw_data["realCount"]):
                complete_content = complete_content.replace("#idiom#", single_raw_data["groundTruth"][single_idiom_index], 1)

            for single_idiom_index in range(single_raw_data["realCount"]):
                single_res_data = {}
                tmp1 = content_splitter(complete_content, single_raw_data["groundTruth"][single_idiom_index], tokenizer)
                idiom_candidates = [idioms_list.index(x) for x in single_raw_data["candidates"][single_idiom_index]]

                single_res_data["sentence_token"] = torch.Tensor(tmp1["sentence_token"])
                single_res_data["sentence_mask"] = torch.Tensor(tmp1["sentence_mask"])
                single_res_data["idiom_mask"] = torch.Tensor(tmp1["idiom_mask"])
                single_res_data["idiom_candidate_index"] = torch.Tensor(idiom_candidates)


                single_res_data["label"] = single_raw_data["candidates"][single_idiom_index].index \
                    (single_raw_data["groundTruth"][single_idiom_index])
            self.data.append(single_res_data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    batch_size = 32
    if_shuffle = True

    myData = IdiomsClassificationDataset("../data/ChID/dev_data.json")
    myDataLoader = DataLoader(myData, batch_size=batch_size, shuffle=if_shuffle)

    print(myDataLoader.dataset.__getitem__(2))



