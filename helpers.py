# ./helpers.py

import re
from typing import List
import numpy as np
import torch

def remove_username(message: str) -> str:
    return re.sub(r'^.+?:\s*', '', message)

def remove_timestamp(message: str) -> str:
    return re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', message)

def rem_user_and_time(message: str) -> str:
    return remove_username(remove_timestamp(message))

def filter_keywords(keywords, min_length=3):
    filtered_keywords = [keyword for keyword in keywords if len(keyword) >= min_length]
    return filtered_keywords

def generate_keyword_groups(keywords: List[str], n: int = 2) -> List[str]:
    return [" ".join(keywords[i:i + n]) for i in range(len(keywords) - n + 1)]

def merge_lists_by_distance(list1, list2, max_new_list_length=500):
    merged_list = []
    i = j = 0
    while i < len(list1) and j < len(list2) and len(merged_list) < max_new_list_length:
        if list1[i][2] < list2[j][2]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1
    while i < len(list1) and len(merged_list) < max_new_list_length:
        merged_list.append(list1[i])
        i += 1
    while j < len(list2) and len(merged_list) < max_new_list_length:
        merged_list.append(list2[j])
        j += 1
    return merged_list

def remove_duplicates(memory_stack):
    memory_dict = {}
    for memory in memory_stack:
        index, _, distance = memory
        if index not in memory_dict or distance < memory_dict[index][2]:
            memory_dict[index] = memory
    return sorted(memory_dict.values(), key=lambda x: x[2])

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


#--------------- Annoy helpers ---------------
def copy_items(src_index, dest_index, num_items):
    for i in range(num_items):
        item = src_index.get_item_vector(i)
        dest_index.add_item(i, item)

#--------------- PyTorch helpers ---------------

def _get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
