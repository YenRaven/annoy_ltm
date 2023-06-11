from math import floor
import time

from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from annoy import AnnoyIndex
from collections import deque
import queue
import concurrent.futures

from extensions.annoy_ltm.helpers import *
from extensions.annoy_ltm.text_preprocessor import TextPreprocessor
from extensions.annoy_ltm.embeddings import generate_embeddings
from extensions.annoy_ltm.keyword_tally import KeywordTally
from extensions.annoy_ltm.annoy_manager import AnnoyManager
from extensions.annoy_ltm.turn_templates import get_turn_templates, apply_turn_templates_to_rows

# parameters which can be customized in settings.json of webui
params = {
    'annoy_output_dir': "extensions/annoy_ltm/outputs/",
    'logger_level': 1, # higher number is more verbose logging. 3 is really as high as any reasonable person should go for normal debugging
    'vector_dim_override': -1, # magic number determined by your loaded model. This parameter is here so that should some style of model in the future not include the hidden_size in the config, this can be used as a workaround.
    'memory_retention_threshold': 0.68, # 0-1, lower value will make memories retain longer but can cause stack to overflow and irrelevant memories to be held onto
    'full_memory_additional_weight': 0.3, # 0-1, smaller value is more weight here.
    'keyword_match_weight': 0.6, # 0-1, smaller value is more weight here.
    'named_entity_match_clamp_min_dist': 0.6, # 0-1, clamp weight to this value, Prevents exact NER match from overriding all other memories. 
    'num_memories_to_retrieve': 5, # the number of related memories to retrieve for the full message and every keyword group and named entity generated from the message. Can cause significant slowdowns.
    'keyword_grouping': 4, # the number to group keywords into. Higher means harder to find an exact match, which makes matches more useful to context but too high and no memories will be returned.
    'keyword_rarity_weight': 1, # Throttles the weight applied to memories favoring unique phrases and vocabularly.
    'maximum_memory_stack_size': 50, # just a cap on the stack so it doesn't blow.
    'prompt_memory_ratio': 0.4 # the ratio of prompt after the character context is applied that will be dedicated for memories.
}

#--------------- Logger ---------------
def logger(msg: str, lvl=5):
    if params['logger_level'] >= lvl:
        print(msg)

#--------------- Custom Prompt Generator ---------------

class ChatGenerator:
    def __init__(self):
        self.memory_stack = deque()
        self.keyword_tally = KeywordTally()
        self.text_preprocessor = TextPreprocessor()
        self.annoy_manager = AnnoyManager(self.text_preprocessor)
        self.annoy_index = None
        
        # Create dictionary for annoy indices
        self.index_to_history_position = {}
        
    #--------------- Memory ---------------
    def compare_text_embeddings(self, text1, text2):
        logger(f"comparing text {text1}\nagainst {text2}", 5)
        text1_embeddings = generate_embeddings(text1, logger=logger)
        text2_embeddings = generate_embeddings(text2, logger=logger)
        logger(f"text1_embeddings: {text1_embeddings}", 6)
        logger(f"text2_embeddings: {text2_embeddings}", 6)
        logger(f"len text1_embeddings: {len(text1_embeddings)}", 6)
        logger(f"len text2_embeddings: {len(text2_embeddings)}", 6)
        cosine_similarity_value = cosine_similarity(text1_embeddings, text2_embeddings)
        logger(f"manually computed cosine similarity: {cosine_similarity_value}", 5)

        return cosine_similarity_value


    def evaluate_memory_relevance(self, state, memory, conversation, min_relevance_threshold=0.2):
        memory_text = ''.join([user_mem + '\n' + bot_mem for user_mem, bot_mem in memory])
        conversation_text = ''.join(conversation)
        logger(f"evaluating memory relevance for memory: {memory}", 4)
        memory_keywords, memory_named_entities = self.text_preprocessor.trim_and_preprocess_text(memory_text, state)
        conversation_keywords, conversation_named_entities = self.text_preprocessor.trim_and_preprocess_text(conversation_text, state)

        memory_keywords = " ".join(filter_keywords(memory_keywords))
        conversation_keywords = " ".join(filter_keywords(conversation_keywords))

        memory_named_entities = " ".join(memory_named_entities)
        conversation_named_entities = " ".join(conversation_named_entities)

        logger(f"comparing memory_keywords against conversation_keywords", 5)
        keyword_similarity_value = self.compare_text_embeddings(memory_keywords, conversation_keywords)

        logger(f"comparing memory_named_entities against conversation_named_entities", 5)
        named_entitiy_similarity_value = self.compare_text_embeddings(memory_named_entities, conversation_named_entities)

        similarity_value = (keyword_similarity_value + named_entitiy_similarity_value) / 2
        logger(f"calculated_similarity: {similarity_value}")
        return similarity_value >= min_relevance_threshold


    def retrieve_related_memories(self, state, annoy_index, input_messages, history_rows, index_to_history_position, keyword_tally, num_related_memories=3, weight=0.5):
        return_memories = set()
        for input_str in input_messages:
            logger(f"retrieving memories for <input> {input_str} </input>", 3)
            if num_related_memories == 0:
                num_related_memories = annoy_index.get_n_items()
            input_embedding = generate_embeddings(remove_username_and_timestamp(input_str, state), logger=logger)
            results_indices = []
            results_distances = []

            # Query for the original input_embedding
            indices, distances = annoy_index.get_nns_by_vector(input_embedding, num_related_memories, include_distances=True)
            results_indices.extend(indices)
            results_distances.extend(map(lambda x: x * weight, distances))
            original_input_results_count = len(results_distances)

            # Get keywords and named entities
            keywords, named_entities = self.text_preprocessor.trim_and_preprocess_text(input_str, state)
            filtered_keywords = filter_keywords(keywords)
            keyword_groups = generate_keyword_groups(filtered_keywords, params['keyword_grouping'])
            logger(f"INPUT_KEYWORDS: {','.join(filtered_keywords)}", 4)

            # Query for each keyword_embedding
            for keyword in keyword_groups:
                keyword_embedding = generate_embeddings(keyword, logger=logger)
                logger(f"looking up keyword \"{keyword}\" embeddings {keyword_embedding}", 5)
                indices, distances = annoy_index.get_nns_by_vector(keyword_embedding, num_related_memories, include_distances=True)
                logger(f"keyword matches: {keyword}\n{indices}\n{distances}", 5)
                results_indices.extend(indices)
                results_distances.extend(map(lambda x: x*params['keyword_match_weight'], distances))

            # Query for each named entity
            named_entities = " ".join(named_entities)
            named_entity_embedding = generate_embeddings(named_entities, logger=logger)
            logger(f"looking up named entity \"{named_entities}\" embeddings {named_entity_embedding}", 5)
            indices, distances = annoy_index.get_nns_by_vector(named_entity_embedding, num_related_memories, include_distances=True)
            logger(f"named_entities matches: {named_entities}\n{indices}\n{distances}", 5)
            results_indices.extend(indices)
            results_distances.extend(map(lambda x: x * (1-params['named_entity_match_clamp_min_dist']) + params['named_entity_match_clamp_min_dist'] , distances))

            if len(results_indices) == 0:
                return [] # If we don't have any results, not much point in progressing.

            # 1. Combine the results
            indices_distances = list(zip(results_indices, results_distances))

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
                usr_keywords, usr_ne = self.text_preprocessor.trim_and_preprocess_text(user_msg, state)
                bot_keywords, bot_ne = self.text_preprocessor.trim_and_preprocess_text(bot_reply, state)
                memory_keywords.extend(filter_keywords(usr_keywords + usr_ne))
                memory_keywords.extend(filter_keywords(bot_keywords + bot_ne))

            significance = params['keyword_rarity_weight'] * keyword_tally.get_significance(memory_keywords)
            logger(f"keywords [{','.join(memory_keywords)}] significance calculated at {significance}", 4)

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



    def build_memory_rows(self, state, history_rows, user_input, max_memory_length, turn_templates, relevance_threshold=0.2):
        user_turn, bot_turn = turn_templates

        # Filter out irrelevant memories
        logger(f"HISTORY_ROWS:{history_rows}", 5)
        conversation = [remove_username_and_timestamp(row, state) for row in history_rows] + [remove_timestamp(user_input)]
        logger(f"CONVERSATION:{conversation}", 5)

        def log_and_check_relevance(memory_tuple, conversation, relevance_threshold):
            relevance_check = self.evaluate_memory_relevance(state, memory_tuple[1], conversation, relevance_threshold)
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


        # Generate annoy database for LTM
        if self.annoy_index == None:
            self.index_to_history_position, self.annoy_index, self.keyword_tally = self.annoy_manager.generate_annoy_db(params, state, self.keyword_tally, logger)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.annoy_manager.generate_annoy_db,
                    params,
                    state,
                    self.keyword_tally,
                    logger
                )
            
            result = None
            while not self.annoy_manager.results_queue.empty():
                try:
                    result = self.annoy_manager.results_queue.get_nowait()
                except queue.Empty:
                    continue  # in case the queue was emptied between the check and get_nowait()

            # Check if a result was actually fetched before trying to unpack it
            if result is not None:
                self.index_to_history_position, self.annoy_index, self.keyword_tally = result

        logger(f"Annoy database has length {self.annoy_index.get_n_items()}", 3)

        # Finding the maximum prompt size
        chat_prompt_size = state['chat_prompt_size']
        max_length = min(get_max_prompt_length(state), chat_prompt_size)

        # Calc the max length for the memory block
        max_memory_length = floor(max_length * params['prompt_memory_ratio']) - len(encode("Memories:\n\n\nChat:\n")[0])

        # Get turn templates
        user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, is_instruct, logger=logger)

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
        related_memories = self.retrieve_related_memories(
            state,
            self.annoy_index,
            memory_trigger,
            history_partial,
            self.index_to_history_position,
            self.keyword_tally,
            num_related_memories=params['num_memories_to_retrieve'],
            weight=params['full_memory_additional_weight']
            )
        
        self.annoy_index.unload() # Unload the index so the next one can save properly

        # Merge new memories into memory stack by distance.
        self.memory_stack = remove_duplicates(merge_memory_lists_by_distance(self.memory_stack, related_memories, max_new_list_length=params['maximum_memory_stack_size']*params['num_memories_to_retrieve']))
        logger(f"merged {len(related_memories)} memories into stack. Stack size:{len(self.memory_stack)}", 3)
        logger(f"MEMORY_STACK:\n{self.memory_stack}", 5)

        memory_rows, num_memories = self.build_memory_rows(state, history_rows[-2:], user_input, max_memory_length, (user_turn, bot_turn), relevance_threshold=params['memory_retention_threshold'])
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
            if len(rows) > 3 + len(memory_rows) + min_rows:                                                       
                rows.pop(3 + len(memory_rows))
            elif len(rows) > 3 + min_rows:
                rows.pop(2)
            else:
                rows.pop(1)

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
