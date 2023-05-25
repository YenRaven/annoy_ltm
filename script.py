from math import floor
import time
from typing import List

import numpy as np
from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from annoy import AnnoyIndex
import torch
import spacy
import re
import hashlib
import json
import os
from collections import deque

# parameters which can be customized in settings.json of webui
params = {
    'annoy_output_dir': "extensions/annoy_ltm/outputs/",
    'logger_level': 1, # higher number is more verbose logging. 3 is really as high as any reasonable person should go for normal debugging
    'vector_dim_override': -1, # magic number determined by your loaded model. This parameter is here so that should some style of model in the future not include the hidden_size in the config, this can be used as a workaround.
    'memory_retention_threshold': 0.68, # 0-1, lower value will make memories retain longer but can cause stack to overflow and irrelevant memories to be held onto
    'full_memory_additional_weight': 0.5, # 0-1, smaller value is more weight here.
    'num_memories_to_retrieve': 5, # the number of related memories to retrieve for the full message and every keyword group generated from the message. Can cause significant slowdowns.
    'keyword_grouping': 4, # the number to group keywords into. Higher means harder to find an exact match, which makes matches more useful to context but too high and no memories will be returned.
    'maximum_memory_stack_size': 50, # just a cap on the stack so it doesn't blow.
    'prompt_memory_ratio': 0.4 # the ratio of prompt after the character context is applied that will be dedicated for memories.
}

#--------------- Helper Functions ---------------
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

#--------------- Hashing and metadata functions ---------------
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
    
    logger(f"Metadata code hash: {metadata['code_hash']}", 5)
    logger(f"Computed code hash: {code_hash}", 5)
    logger(f"Metadata messages hash: {metadata['messages_hash']}", 5)
    logger(f"Computed messages hash: {messages_hash}", 5)
    
    if metadata['code_hash'] != code_hash:
        return False

    if metadata['messages_hash'] != messages_hash:
        return False

    return True

#--------------- Annoy helpers ---------------
def copy_items(src_index, dest_index, num_items):
    for i in range(num_items):
        item = src_index.get_item_vector(i)
        dest_index.add_item(i, item)

#--------------- PyTorch helpers ---------------

def _get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--------------- Embeddings ---------------
def generate_embeddings(text):
    input_ids = shared.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(_get_device())  # Move input_ids to the model's device
    input_ids = input_ids.long() # ensure the values are not floats

    with torch.no_grad():
        input_embeds = shared.model.model.embed_tokens(input_ids)

    input_embeds = input_embeds.mean(dim=1).squeeze(0)  # Remove the extra dimension
    result = input_embeds.cpu().numpy().flatten()  # Convert to NumPy array and flatten
    logger(f"generating embeddings for text: {text}\n{result}", 5)
    return result

#--------------- Hidden Size Helper -------------
def _get_hidden_size():
    if params['vector_dim_override'] != -1:
        return params['vector_dim_override']
    
    try:
        return shared.model.model.config.hidden_size
    except AttributeError:
        return len(generate_embeddings('generate a set of embeddings to determin size of result list'))

#--------------- Turn Templates ---------------
def get_turn_templates(state, is_instruct):

    
    logger(f"state['turn_template']: {state['turn_template']}", 5)
    
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
    logger(f"turn_template replacements: {replacements}", 5)

    user_turn = replace_all(template.split('<|bot|>')[0], replacements)
    bot_turn = replace_all('<|bot|>' + template.split('<|bot|>')[1], replacements)
    user_turn_stripped = replace_all(user_turn.split('<|user-message|>')[0], replacements)
    bot_turn_stripped = replace_all(bot_turn.split('<|bot-message|>')[0], replacements)

    logger(f"turn_templates:\nuser_turn:{user_turn}\nbot_turn:{bot_turn}\nuser_turn_stripped:{user_turn_stripped}\nbot_turn_stripped:{bot_turn_stripped}", 5)

    return user_turn, bot_turn, user_turn_stripped, bot_turn_stripped

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

#--------------- Keywords ---------------
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

    return keywords
    
#--------------- Memory ---------------
def evaluate_memory_relevance(memory, conversation, min_relevance_threshold=0.2):
    logger(f"evaluating memory relevance for memory: {memory}", 4)
    memory_keywords = " ".join(filter_keywords(preprocess_and_extract_keywords(''.join([user_mem + '\n' + bot_mem for user_mem, bot_mem in memory]))))
    conversation_keywords = " ".join(filter_keywords(preprocess_and_extract_keywords(''.join(conversation))))
    logger(f"comparing keywords {memory_keywords}\nagainst conversation {conversation_keywords}", 5)
    memory_embeddings = generate_embeddings(memory_keywords)
    conversation_embeddings = generate_embeddings(conversation_keywords)
    logger(f"memory_embeddings: {memory_embeddings}", 6)
    logger(f"conversation_embeddings: {conversation_embeddings}", 6)
    logger(f"len memory_embeddings: {len(memory_embeddings)}", 6)
    logger(f"len conversation_embeddings: {len(conversation_embeddings)}", 6)
    cosine_similarity_value = cosine_similarity(memory_embeddings, conversation_embeddings)
    logger(f"manually computed cosine similarity: {cosine_similarity_value}", 5)
    return cosine_similarity_value >= min_relevance_threshold


def retrieve_related_memories(annoy_index, input_messages, history_rows, index_to_history_position, keyword_tally, num_related_memories=3, weight=0.5):
    return_memories = set()
    for input_str in input_messages:
        logger(f"retrieving memories for <input> {input_str} </input>", 3)
        if num_related_memories == 0:
            num_related_memories = annoy_index.get_n_items()
        input_embedding = generate_embeddings(rem_user_and_time(input_str))
        results_indices = []
        results_distances = []

        # Query for the original input_embedding
        indices, distances = annoy_index.get_nns_by_vector(input_embedding, num_related_memories, include_distances=True)
        results_indices.extend(indices)
        results_distances.extend(distances)
        original_input_results_count = len(results_distances)

        # Get keywords
        keywords = preprocess_and_extract_keywords(input_str)
        filtered_keywords = filter_keywords(keywords)
        keyword_groups = generate_keyword_groups(filtered_keywords, params['keyword_grouping'])
        logger(f"INPUT_KEYWORDS: {','.join(filtered_keywords)}", 4)

        # Query for each keyword_embedding
        for keyword in keyword_groups:
            keyword_embedding = generate_embeddings(keyword)
            logger(f"looking up keyword \"{keyword}\" embeddings {keyword_embedding}", 5)
            indices, distances = annoy_index.get_nns_by_vector(keyword_embedding, num_related_memories, include_distances=True)
            logger(f"keyword matches: {keyword}\n{indices}\n{distances}", 5)
            results_indices.extend(indices)
            results_distances.extend(distances)

        if len(results_indices) == 0:
            return [] # If we don't have any results, not much point in progressing.

        # 1. Combine the results
        indices_distances = list(zip(results_indices, results_distances))

        # 2. Apply the weight to the original input distances
        for i in range(original_input_results_count):
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

        return_memories.update(set(weighted_history_positions))
        # return_memories.extend(weighted_history_positions)

    # 4. Get the related memories using the new sorted list
    related_memories = [(pos, shared.history['internal'][max(0, pos - 1):pos + 1], distance) for pos, distance in list(return_memories)]

    # Get keywords for each memory and calculate their significance
    for i in range(len(related_memories)):
        index, memory, distance = related_memories[i]
        memory_keywords = []
        for user_msg, bot_reply in memory:
            memory_keywords.extend(preprocess_and_extract_keywords(user_msg))
            memory_keywords.extend(preprocess_and_extract_keywords(bot_reply))

        significance = keyword_tally.get_significance(memory_keywords)

        # Apply the significance ratio to the memory's distance value
        related_memories[i] = (index, memory, distance * significance)

    # 5. Sort the new list
    sorted_weighted_related_memories = sorted(related_memories, key=lambda x: (x[2], x[0]))
    logger(f"RESULTS: {sorted_weighted_related_memories}", 4)

    # 6. Filter out memories that are already present in the history added to the prompt
    non_duplicate_memories = [
        (index, memory, distance) for index, memory, distance in sorted_weighted_related_memories
        if all(msg not in history_rows for msg in memory)
    ]

    return non_duplicate_memories

class KeywordTally:
    def __init__(self):
        self.keyword_tally_count = {}
        self.total_keywords = 0

    def tally(self, keywords):
        for keyword in keywords:
            self.total_keywords += 1
            if keyword in self.keyword_tally_count:
                self.keyword_tally_count[keyword] += 1
            else:
                self.keyword_tally_count[keyword] = 1

    def get_significance(self, keywords):
        significance = 0
        for keyword in keywords:
            if keyword in self.keyword_tally_count:
                ratio = self.keyword_tally_count[keyword] / self.total_keywords
                significance += 1 - ratio
        return significance / len(keywords)
    
    def exportKeywordTally(self):
        return self.keyword_tally_count

    def importKeywordTally(self, keyword_tally_data):
        self.keyword_tally_count = keyword_tally_data
        self.total_keywords = sum(keyword_tally_data.values())



#--------------- Custom Prompt Generator ---------------

class ChatGenerator:
    def __init__(self):
        self.memory_stack = deque()
        self.keyword_tally = KeywordTally()

    def build_memory_rows(self, history_rows, user_input, max_memory_length, turn_templates, relevance_threshold=0.2):
        user_turn, bot_turn = turn_templates

        # Filter out irrelevant memories
        logger(f"HISTORY_ROWS:{history_rows}", 5)
        conversation = [rem_user_and_time(row) for row in history_rows] + [remove_timestamp(user_input)]
        logger(f"CONVERSATION:{conversation}", 5)

        def log_and_check_relevance(memory_tuple, conversation, relevance_threshold):
            relevance_check = evaluate_memory_relevance(memory_tuple[1], conversation, relevance_threshold)
            logger(f"\nrelevance_check: {relevance_check}\nmemory_tuple: {memory_tuple}", 4)
            return relevance_check

        # Use the log_and_check_relevance function in the list comprehension
        new_memory_stack = [memory_tuple for memory_tuple in self.memory_stack if log_and_check_relevance(memory_tuple, conversation, relevance_threshold)]
        new_memory_stack = new_memory_stack[params['maximum_memory_stack_size']:]

        logger(f"MEMORY_STACK:{new_memory_stack}", 5)
        logger(f"MEMORY_STACK SIZE: {len(new_memory_stack)}", 3)

        # Create memory_rows

        memory_len = 0
        memory_index = 0
        returned_memories = 0
        memory_rows = []
        last_index = 0
        last_memory_rows_count = 0

        while memory_index < len(new_memory_stack) and memory_len < max_memory_length:
            index, memory, _ = new_memory_stack[memory_index]
            new_memory_len = memory_len

            i = len(memory) - 1
            stop_i = 0
            new_memory_rows_count = 0
            if last_index == index-1:
                i -= 1 # should this happen to be a continuation, then we will skip adding the duplicate memory
            if last_index == index+1:
                stop_i = 1
            while i >= stop_i and memory_len < max_memory_length:

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
                    if last_index == index+1:
                        memory_rows.insert(last_memory_rows_count, proposed_bot_turn)
                        memory_rows.insert(last_memory_rows_count, proposed_user_turn)
                    else:
                        memory_rows.insert(0, proposed_bot_turn)
                        memory_rows.insert(0, proposed_user_turn)
                    
                    logger(f"adding memory rows from stack for index {index}...\n{proposed_user_turn}\n{proposed_bot_turn}\n", 3)
                    new_memory_rows_count += 2
                        
                else:
                    break

                i -= 1

            memory_len = new_memory_len
            returned_memories += 1
            memory_index += 1
            last_index = index
            last_memory_rows_count = new_memory_rows_count

        non_relavent_memories = [(index, memory) for index, memory, _ in self.memory_stack if index not in [i for i, _, _ in new_memory_stack]]
        memory_index = 0
        while memory_index < len(non_relavent_memories) and memory_len < max_memory_length :
            index, memory = non_relavent_memories[memory_index]
            new_memory_len = memory_len

            i = len(memory) - 1
            stop_i = 0
            new_memory_rows_count = 0
            if last_index == index-1:
                i -= 1 # should this happen to be a continuation, then we will skip adding the duplicate memory
            if last_index == index+1:
                stop_i = 1
            while i >= stop_i and memory_len < max_memory_length:
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
                    if last_index == index+1:
                        memory_rows.insert(last_memory_rows_count, proposed_bot_turn)
                        memory_rows.insert(last_memory_rows_count, proposed_user_turn)
                    else:
                        memory_rows.insert(0, proposed_bot_turn)
                        memory_rows.insert(0, proposed_user_turn)
                    
                    logger(f"adding memory rows from non_relavant for index {index}...\n{proposed_user_turn}\n{proposed_bot_turn}\n", 3)
                    new_memory_rows_count += 2
                
                else:
                    break


                i -= 1
            
            memory_len = new_memory_len
            returned_memories += 1
            memory_index += 1
            last_index = index
            last_memory_rows_count = new_memory_rows_count

        self.memory_stack = new_memory_stack
            
        return memory_rows, returned_memories

    def custom_generate_chat_prompt(self, user_input, state, **kwargs):
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

        metadata_file = f"{params['annoy_output_dir']}{shared.character}-annoy-metadata.json"
        annoy_index_file = f"{params['annoy_output_dir']}{shared.character}-annoy_index.ann"

        metadata = load_metadata(metadata_file)
        if metadata == None:
            logger(f"failed to load character annoy metadata, generating from scratch...", 1)
        else:
            logger(f"loaded metadata file ({len(metadata['messages_hash'])})", 2)

        hidden_size = _get_hidden_size()

        loaded_annoy_index = AnnoyIndex(hidden_size, 'angular')
        loaded_history_last_index = 0
        
        annoy_index = AnnoyIndex(hidden_size, 'angular')
        
        if check_hashes(metadata):
            loaded_annoy_index.load(annoy_index_file)
            loaded_history_items = loaded_annoy_index.get_n_items()
            if loaded_history_items < 1:
                logger(f"hashes check passed but no items found in annoy db. rebuilding annoy db...", 2)
            else:
                logger(f"hashes check passed, proceeding to load existing memory db...", 2)
                self.keyword_tally.importKeywordTally(metadata['keyword_tally'])
                index_to_history_position = {int(k): v for k, v in metadata['index_to_history_position'].items()}
                loaded_history_last_index = index_to_history_position[loaded_history_items-1]
                logger(f"loaded {loaded_history_last_index} items from existing memory db", 3)
                copy_items(loaded_annoy_index, annoy_index, loaded_history_items)
                loaded_annoy_index.unload()
        else:
            logger(f"hashes check failed, either an existing message changed unexpectdly or the extension code has changed. Rebuilding annoy db...", 2)
            self.keyword_tally = KeywordTally()

        formated_history_rows = apply_turn_templates_to_rows(shared.history['internal'][loaded_history_last_index:], state)
        logger(f"found {len(formated_history_rows)} rows of chat history to be added to memory db. adding items...", 3)
        unique_index = len(index_to_history_position)
        for i, row in enumerate(formated_history_rows):
            for msg in row:
                trimmed_msg = rem_user_and_time(msg)
                if trimmed_msg and len(trimmed_msg) > 0:
                    # Add the full message
                    logger(f"HISTORY_{i+1}_MSG: {msg}", 4)
                    embeddings = generate_embeddings(trimmed_msg)
                    annoy_index.add_item(unique_index, embeddings)
                    index_to_history_position[unique_index] = i+loaded_history_last_index
                    unique_index += 1
                
                    # Add keywords
                    keywords = preprocess_and_extract_keywords(msg)
                    self.keyword_tally.tally(keywords) # Keep a tally of all keywords
                    filtered_keywords = filter_keywords(keywords)
                    keyword_groups = generate_keyword_groups(filtered_keywords, params['keyword_grouping'])
                    logger(f"HISTORY_{i+1}_KEYWORDS: {','.join(filtered_keywords)}", 4)
                    for keyword in keyword_groups:
                        embeddings = generate_embeddings(keyword)
                        logger(f"storing keyword \"{keyword}\" with embeddings {embeddings}", 5)
                        annoy_index.add_item(unique_index, embeddings)
                        index_to_history_position[unique_index] = i+loaded_history_last_index
                        unique_index += 1

        annoy_index.build(10)
        
        # Save the annoy index and metadata
        code_hash, messages_hash = compute_hashes()
        metadata = {'code_hash': code_hash, 'messages_hash': messages_hash, 'index_to_history_position': index_to_history_position, 'keyword_tally': self.keyword_tally.exportKeywordTally()}
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
        max_memory_length = floor(max_length * params['prompt_memory_ratio']) - len(encode("Memories:\n\n\nChat:\n")[0])

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
        memory_trigger = []
        if len(shared.history['internal']) > 0 and len(shared.history['internal'][-1]) > 1:
            memory_trigger.append(shared.history['internal'][-1][1])
        memory_trigger.append(user_input)
        related_memories = retrieve_related_memories(
            annoy_index,
            memory_trigger,
            history_partial,
            index_to_history_position,
            self.keyword_tally,
            num_related_memories=params['num_memories_to_retrieve'],
            weight=params['full_memory_additional_weight']
            )

        # Merge new memories into memory stack by distance.
        self.memory_stack = remove_duplicates(merge_lists_by_distance(self.memory_stack, related_memories, max_new_list_length=params['maximum_memory_stack_size']*params['num_memories_to_retrieve']))
        logger(f"merged {len(related_memories)} memories into stack. Stack size:{len(self.memory_stack)}", 3)
        logger(f"MEMORY_STACK:\n{self.memory_stack}", 5)

        memory_rows, num_memories = self.build_memory_rows(history_rows[-2:], user_input, max_memory_length, (user_turn, bot_turn), relevance_threshold=params['memory_retention_threshold'])
        logger(f"memory_rows:\n{memory_rows}", 5)
        # Remove the least relevant memory row from the memory stack so that the stack will be worked through one memory at a time with each prompt.
        if num_memories > 0 and len(self.memory_stack) > 0:
            self.memory_stack.pop(min(len(self.memory_stack)-1, num_memories-1))
        

        # Insert memory_rows to the prompt
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
        
generator = ChatGenerator()

def custom_generate_chat_prompt(user_input, state, **kwargs):
    return generator.custom_generate_chat_prompt(user_input, state, **kwargs)
