from math import floor
import time
from typing import List
from modules.html_generator import fix_newlines
from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, decode, generate_reply, get_max_prompt_length
from annoy import AnnoyIndex
import torch
import spacy
from spacy.matcher import Matcher
import re
import hashlib
import json
import os


# parameters which can be customized in settings.json of webui
params = {
    'logger_level': 2,
    'vector_dim': 6656
}

#--------------- Helper Functions ---------------------
def logger(msg: str, lvl=5):
    if params['logger_level'] >= lvl:
        print(msg)

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

# Hashing and metadata functions
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

def compute_hashes(remove_last_user_message=False):
    code_hash = compute_file_hash(__file__)
    if remove_last_user_message:
        messages_hash = [hashlib.md5(str(msg).encode()).hexdigest() for msg in shared.history['internal'][:-1]]
    else:
        messages_hash = [hashlib.md5(str(msg).encode()).hexdigest() for msg in shared.history['internal']]
    return code_hash, messages_hash

def check_hashes(metadata):
    if metadata is None:
        return False

    code_hash, messages_hash = compute_hashes(remove_last_user_message=True)
    
    logger(f"Metadata code hash: {metadata['code_hash']}", 1)
    logger(f"Computed code hash: {code_hash}", 1)
    logger(f"Metadata messages hash: {metadata['messages_hash']}", 1)
    logger(f"Computed messages hash: {messages_hash}", 1)
    
    if metadata['code_hash'] != code_hash:
        return False

    if metadata['messages_hash'] != messages_hash:
        return False

    return True

# Annoy helpers
def copy_items(src_index, dest_index, num_items):
    for i in range(num_items):
        item = src_index.get_item_vector(i)
        dest_index.add_item(i, item)

# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text

def _get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings(text):
    input_ids = shared.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(_get_device())  # Move input_ids to the model's device

    with torch.no_grad():
        input_embeds = shared.model.model.embed_tokens(input_ids)

    input_embeds = input_embeds.mean(dim=1).squeeze(0)  # Remove the extra dimension
    result = input_embeds.cpu().numpy().flatten()  # Convert to NumPy array and flatten
    logger(f"generating embeddings for text: {text}\n{result}", 5)
    return result

def get_turn_templates(state, is_instruct):
    
    # Building the turn templates
    if 'turn_template' not in state or state['turn_template'] == '':
        if is_instruct:
            template = '<|user|>\n<|user-message|>\n<|bot|>\n<|bot-message|>\n'
        else:
            template = '<|user|>: <|user-message|>\n<|bot|>: <|bot-message|>\n'
    else:
        template = state['turn_template'].replace(r'\n', '\n')

    replacements = {
        '<|user|>': state['name1'].strip(),
        '<|bot|>': state['name2'].strip(),
    }

    user_turn = replace_all(template.split('<|bot|>')[0], replacements)
    bot_turn = replace_all('<|bot|>' + template.split('<|bot|>')[1], replacements)
    user_turn_stripped = replace_all(user_turn.split('<|user-message|>')[0], replacements)
    bot_turn_stripped = replace_all(bot_turn.split('<|bot-message|>')[0], replacements)

    return user_turn, bot_turn, user_turn_stripped, bot_turn_stripped


nlp = spacy.load("en_core_web_sm", disable=["parser"])

def preprocess_and_extract_keywords(text):
    text_to_process = rem_user_and_time(text)
    # Tokenization, lowercasing, and stopword removal
    tokens = [token.text.lower() for token in nlp(text_to_process) if not token.is_stop]

    # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

    # Named Entity Recognition
    doc = nlp(text_to_process)
    named_entities = [ent.text for ent in doc.ents]

    keywords = lemmatized_tokens + named_entities

    return generate_keyword_groups(filter_keywords(keywords), 3)

def apply_turn_templates_to_rows(rows, state):
    is_instruct = state['mode'] == 'instruct'
    user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, is_instruct)
    output_rows = []
    for i, row in enumerate(rows):
        if row[0] not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            user_row = replace_all(user_turn, {'<|user-message|>': row[0].strip(), '<|round|>': str(i)})
        else:
            user_row = row[0]
        bot_row = bot_turn.replace('<|bot-message|>', row[1].strip())
        output_rows.append((user_row, bot_row))

    return output_rows

def retrieve_related_memories(annoy_index, input_str, history_rows, index_to_history_position, num_related_memories=10, weight=0.5):
    if num_related_memories == 0:
        num_related_memories = annoy_index.get_n_items()
    input_embedding = generate_embeddings(rem_user_and_time(input_str))
    results_indices = []
    results_distances = []

    # Query for the original input_embedding
    indices, distances = annoy_index.get_nns_by_vector(input_embedding, num_related_memories, include_distances=True)
    results_indices.extend(indices)
    results_distances.extend(distances)

    # Get keywords
    keywords = preprocess_and_extract_keywords(input_str)
    logger(f"INPUT_KEYWORDS: {','.join(keywords)}", 4)

    # Query for each keyword_embedding
    for keyword in keywords:
        keyword_embedding = generate_embeddings(keyword)
        logger(f"looking up keyword \"{keyword}\" embeddings {keyword_embedding}", 5)
        indices, distances = annoy_index.get_nns_by_vector(keyword_embedding, num_related_memories, include_distances=True)
        logger(f"keyword matches: {keyword}\n{indices}\n{distances}", 5)
        results_indices.extend(indices)
        results_distances.extend(distances)

    # 1. Combine the results
    indices_distances = list(zip(results_indices, results_distances))

    # 2. Apply the weight to the original input distances
    for i in range(num_related_memories):
        indices_distances[i] = (indices_distances[i][0], indices_distances[i][1] * weight)

    # 3. Create a new list of unique history positions tupled with their distance while applying weights for duplicates
    history_positions_distances = {}
    for index, distance in indices_distances:
        history_position = index_to_history_position[index]
        if history_position in history_positions_distances:
            history_positions_distances[history_position].append(distance)
        else:
            history_positions_distances[history_position] = [distance]

    weighted_history_positions = [(pos, min(distances) / len(distances)) for pos, distances in history_positions_distances.items()]

    # 4. Sort the new list
    sorted_weighted_history_positions = sorted(weighted_history_positions, key=lambda x: x[1])

    logger(f"RESULTS: {sorted_weighted_history_positions}", 4)

    # 5. Get the related memories using the new sorted list
    related_memories = [(pos, shared.history['internal'][max(0, pos - 1):pos + 1]) for pos, _ in sorted_weighted_history_positions]

    # 6. Filter out memories that are already present in the history added to the prompt
    non_duplicate_memories = [
        (index, memory) for index, memory in related_memories
        if all(msg not in history_rows for msg in memory)
    ]

    return non_duplicate_memories


def custom_generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    is_instruct = state['mode'] == 'instruct'
    rows = [state['context'] if is_instruct else f"{state['context'].strip()}\n"]
    min_rows = 3

    # Create dictionary for annoy indices
    index_to_history_position = {}

    # Generate annoy database for LTM
    start_time = time.time()

    metadata_file = f"annoy/{shared.character}-annoy-metadata.json"
    annoy_index_file = f"annoy/{shared.character}-annoy_index.ann"

    metadata = load_metadata(metadata_file)
    if metadata == None:
        logger(f"failed to load character annoy metadata, generating from scratch...", 1)
    else:
        logger(f"loaded metadata file ({len(metadata['messages_hash'])})", 2)

    loaded_annoy_index = AnnoyIndex(params['vector_dim'], 'angular')
    loaded_history_last_index = 0
    
    annoy_index = AnnoyIndex(params['vector_dim'], 'angular')
    
    if check_hashes(metadata):
        logger(f"hashes check passed, proceeding to load existing memory db...", 2)
        index_to_history_position = {int(k): v for k, v in metadata['index_to_history_position'].items()}
        loaded_annoy_index.load(annoy_index_file)
        loaded_history_items = loaded_annoy_index.get_n_items()
        loaded_history_last_index = index_to_history_position[loaded_history_items-1]
        logger(f"loaded {loaded_history_last_index} items from existing memory db", 3)
        copy_items(loaded_annoy_index, annoy_index, loaded_history_items)
        loaded_annoy_index.unload()
    else:
        logger(f"hashes check failed, either an existing message changed unexpectdly or the extension code has changed. Rebuilding annoy db...", 2)

    formated_history_rows = apply_turn_templates_to_rows(shared.history['internal'][loaded_history_last_index:], state)
    logger(f"found {len(formated_history_rows)} rows of chat history to be added to memory db. adding items...", 3)
    unique_index = len(index_to_history_position)
    for i, row in enumerate(formated_history_rows):
        for msg in row:
            # Add the full message
            logger(f"HISTORY_{i+1}_MSG: {msg}", 4)
            embeddings = generate_embeddings(rem_user_and_time(msg))
            annoy_index.add_item(unique_index, embeddings)
            index_to_history_position[unique_index] = i+loaded_history_last_index
            unique_index += 1

            # Add keywords
            keywords = preprocess_and_extract_keywords(msg)
            logger(f"HISTORY_{i+1}_KEYWORDS: {','.join(keywords)}", 4)
            for keyword in keywords:
                embeddings = generate_embeddings(keyword)
                logger(f"storing keyword \"{keyword}\" with embeddings {embeddings}", 5)
                annoy_index.add_item(unique_index, embeddings)
                index_to_history_position[unique_index] = i+loaded_history_last_index
                unique_index += 1

    annoy_index.build(10)
    
    # Save the annoy index and metadata
    code_hash, messages_hash = compute_hashes()
    metadata = {'code_hash': code_hash, 'messages_hash': messages_hash, 'index_to_history_position': index_to_history_position}
    save_metadata(metadata, metadata_file)
    annoy_index.save(annoy_index_file)
    
    end_time = time.time()
    logger(f"building annoy index took {end_time-start_time} seconds...", 1)

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]

    max_length = min(get_max_prompt_length(state), chat_prompt_size)
    # Calc the max length for the memory block
    max_memory_length = floor(max_length * 0.3) - len(encode("Memories:\n\n\nChat:\n")[0])

    # Get turn templates
    user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, is_instruct)

    # Building the prompt
    memories_header = "\nMemories:\n"
    chat_header = "\nChat:\n"
    mem_head_len = len(encode(memories_header)[0])
    chat_head_len = len(encode(chat_header)[0])
    history_partial = []
    history_rows = []
    i = len(shared.history['internal']) - 1
    max_history_length = max_length - len(encode(''.join(rows))[0]) - max_memory_length - mem_head_len - chat_head_len
    while i >= 0 and len(encode(''.join(history_rows))[0]) < max_history_length:
        if _continue and i == len(shared.history['internal']) - 1:
            history_rows.insert(0, bot_turn_stripped + shared.history['internal'][i][1].strip())
        else:
            history_rows.insert(0, bot_turn.replace('<|bot-message|>', shared.history['internal'][i][1].strip()))

        string = shared.history['internal'][i][0]
        if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            history_rows.insert(0, replace_all(user_turn, {'<|user-message|>': string.strip(), '<|round|>': str(i)}))

        history_partial.append(shared.history['internal'][i])
        i -= 1

    # Adding related memories to the prompt
    rows.append(memories_header)
    related_memories = retrieve_related_memories(annoy_index, user_input, history_partial, index_to_history_position)

    memory_len = 0
    memory_index = 0
    memory_rows = []

    while memory_index < len(related_memories):
        index, memory = related_memories[memory_index]
        new_memory_len = memory_len

        i = len(memory) - 1
        while i >= 0 and memory_len < max_memory_length:
            turn = memory[i]
            proposed_user_turn = ''
            proposed_bot_turn = ''

            if len(turn) > 0:
                user_memory, ai_memory = turn
                logger(f"user_memory:{user_memory}\nai_memory:{ai_memory}", 5)
                proposed_user_turn = replace_all(user_turn, {'<|user-message|>': user_memory.strip(), '<|round|>': str(index)})
                proposed_bot_turn = bot_turn.replace('<|bot-message|>', ai_memory.strip())
                
                new_memory_len = new_memory_len + len(encode(proposed_user_turn)[0]) + len(encode(proposed_bot_turn)[0])

            if new_memory_len <= max_memory_length:
                memory_rows.insert(0, proposed_bot_turn)
                memory_rows.insert(0, proposed_user_turn)
            else:
                break

            i -= 1

        memory_len = new_memory_len
        memory_index += 1
    
    rows.extend(memory_rows)
    rows.append(chat_header)

    # Insert the history_rows
    rows.extend(history_rows)

    if impersonate:
        min_rows = 2
        rows.append(user_turn_stripped.rstrip(' '))
    elif not _continue:
        # Adding the user message
        if len(user_input) > 0:
            rows.append(replace_all(user_turn, {'<|user-message|>': user_input.strip(), '<|round|>': str(len(shared.history["internal"]))}))

        # Adding the Character prefix
        rows.append(apply_extensions("bot_prefix", bot_turn_stripped.rstrip(' ')))

    while len(rows) > min_rows and len(encode(''.join(rows))[0]) >= max_length:
        rows.pop(3 + len(memory_rows))

    prompt = ''.join(rows)
    logger(f"custom_generated_prompt:\n\n{prompt}\n\n", 2)
    logger(f"prompt_len:{len(encode(prompt)[0])}\nmax_length:{max_length}\nmax_memory_length:{max_memory_length}\nmax_history_length:{max_history_length}\nmax_content_length:{max_history_length+max_memory_length}\ntotal_content_length:{len(encode(rows[0])[0]) + max_history_length + max_memory_length}", 2)

    if also_return_rows:
        return prompt, rows
    else:
        return prompt
