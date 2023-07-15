# ./metadata.py

from modules import shared

import hashlib
import json
import os
import glob

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_metadata(metadata, filepath):
    dir_path = os.path.dirname(filepath)

    # Check if the directory exists, and create it if necessary
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(filepath, 'w') as f:
        json.dump(metadata, f)

def load_metadata(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def compute_hashes(history, remove_last_user_message=False):
    # Compute hash for each Python file in the same directory
    python_files = glob.glob(os.path.join(os.path.dirname(__file__), '*.py'))
    code_hash = ''.join(sorted([compute_file_hash(file) for file in python_files]))

    if remove_last_user_message:
        messages_hash = [hashlib.md5(str(msg).encode()).hexdigest() for msg in history['internal'][:-1]]
    else:
        messages_hash = [hashlib.md5(str(msg).encode()).hexdigest() for msg in history['internal']]

    return code_hash, messages_hash


def check_hashes(metadata, history, logger):
    if metadata is None:
        return False

    code_hash, messages_hash = compute_hashes(history, remove_last_user_message=True)
    
    logger(f"Metadata code hash: {metadata['code_hash']}", 5)
    logger(f"Computed code hash: {code_hash}", 5)
    logger(f"Metadata messages hash: {metadata['messages_hash']}", 5)
    logger(f"Computed messages hash: {messages_hash}", 5)
    
    if metadata['code_hash'] != code_hash:
        return False

    if metadata['messages_hash'] != messages_hash:
        return False
    
    if metadata['model_name'] != shared.model_name:
        return False

    return True