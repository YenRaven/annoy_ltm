# ./helpers.py

import re
from typing import List
import numpy as np
import torch

def remove_username(message: str) -> str:
    """
    Removes the username prefix from a message string. Assumes the username is at the start of the message followed by ': '. Returns the message without the username.
    """
    return re.sub(r'^.+?:\s*', '', message)

def remove_timestamp(message: str) -> str:
    """
    Removes timestamp from a message string. Assumes timestamp in format 'YYYY-MM-DD HH:MM:SS'. Returns the message without the timestamp.
    """
    return re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', message)

def remove_username_and_timestamp(message: str) -> str:
    """
    Removes both the username and timestamp from a message string. Returns the message without the username and timestamp.
    """
    return remove_username(remove_timestamp(message))

def filter_keywords(keywords, min_length=3):
    """
    Filters a list of keywords, removing any keywords shorter than min_length. Returns the filtered list of keywords.
    """
    filtered_keywords = [keyword for keyword in keywords if len(keyword) >= min_length]
    return filtered_keywords

def generate_keyword_groups(keywords: List[str], n: int = 2) -> List[str]:
    """
    Generates groups of n consecutive keywords from a list. Returns a list of keyword groups.
    """
    return [" ".join(keywords[i:i + n]) for i in range(len(keywords) - n + 1)]

def merge_memory_lists_by_distance(list1, list2, max_new_list_length=500):
    """
    Merges two memory lists by their distance values. Returns a merged list up to a maximum length of max_new_list_length.
    """
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
    """
    Removes duplicate memories from a memory stack, keeping the one with the shortest distance. Returns a list of unique memories sorted by distance.
    """
    memory_dict = {}
    for memory in memory_stack:
        index, _, distance = memory
        if index not in memory_dict or distance < memory_dict[index][2]:
            memory_dict[index] = memory
    return sorted(memory_dict.values(), key=lambda x: x[2])

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors a and b
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Replace multiple string pairs in a string
def replace_all(text, dic):
    """
    Replaces all instances of certain substrings in a text string.
    """
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


#--------------- Annoy helpers ---------------
def copy_items(src_index, dest_index, num_items):
    """
    Copies all items from one annoy index to another
    """
    for i in range(num_items):
        item = src_index.get_item_vector(i)
        dest_index.add_item(i, item)

#--------------- PyTorch helpers ---------------

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
