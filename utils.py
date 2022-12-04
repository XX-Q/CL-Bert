from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import torch


def content_splitter(content, idiom, tokenizer, length=300):
    """
    split content into setted length, generate tokens and masks

    content: single complete content
    idiom: single idiom

    """
    res_dict = {}
    idiom_index = content.find(idiom)
    splitted_content = ""
    # print(len(content))
    if idiom_index < int(length/2):
        splitted_content = content[:length]
    else:
        splitted_content = content[idiom_index-int(length/2)+1:idiom_index+int(length/2)-1]
    splitted_idiom_index = splitted_content.find(idiom)
    sentence_token = np.array(tokenizer.encode(splitted_content)[:300])
    sentence_mask = np.ones(len(sentence_token), dtype=int)
    idiom_mask = np.zeros(len(sentence_token), dtype=int)
    idiom_mask[splitted_idiom_index:splitted_idiom_index+4] = 1
    
    sentence_token = np.pad(sentence_token, (0,length-len(sentence_token)), 'constant', constant_values=tokenizer.pad_token_id)
    sentence_mask = np.pad(sentence_mask, (0,length-len(sentence_mask)), 'constant', constant_values=tokenizer.pad_token_id)
    idiom_mask = np.pad(idiom_mask, (0,length-len(idiom_mask)), 'constant', constant_values=tokenizer.pad_token_id)

    res_dict["sentence_token"] = sentence_token
    res_dict["sentence_mask"] = sentence_mask
    res_dict["idiom_mask"] = idiom_mask

    return res_dict
    
    


def idiom_candidate_splitter(idiom_candidates, tokenizer, length=300):
    """
    split the idiom and explanation into setted length, generate tokens and masks
    """

    res_dict = {}
    idiom_candidate_patterns_token = []
    idiom_candidate_patterns_mask = []
    for idiom_candidate in idiom_candidates:
        idiom_candidate_pattern_token = np.array(tokenizer.encode(idiom_candidate)[:length])
        idiom_candidate_pattern_mask = np.ones(len(idiom_candidate_pattern_token), dtype=int)
        idiom_candidate_pattern_token = np.pad(idiom_candidate_pattern_token, (0,length-len(idiom_candidate_pattern_token)), 'constant', constant_values=tokenizer.pad_token_id)
        idiom_candidate_pattern_mask = np.pad(idiom_candidate_pattern_mask, (0,length-len(idiom_candidate_pattern_mask)), 'constant', constant_values=tokenizer.pad_token_id)

        idiom_candidate_patterns_token.append(idiom_candidate_pattern_token)
        idiom_candidate_patterns_mask.append(idiom_candidate_pattern_mask)
    
    res_dict["idiom_candidate_patterns_token"] = np.array(idiom_candidate_patterns_token)
    res_dict["idiom_candidate_patterns_mask"] = np.array(idiom_candidate_patterns_mask)
    return res_dict
    