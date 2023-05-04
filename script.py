import asyncio
import atexit
from math import floor
import re
from modules.html_generator import fix_newlines
import modules.shared
from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, decode, generate_reply, get_max_prompt_length
from annoy import AnnoyIndex
import torch
import os
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

def run_async(coro_func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro_func(*args, **kwargs))
    finally:
        loop.close()


# Parameters
VECTOR_DIM = 6656  # Set this based on your needs
DATABASE_PATH = 'annoy_db.ann'
params = {
    "add_all_images_to_prompt": False,
    # device to run CLIP on
    "clip_device": None,
    # bits to load clip in either 32 or 16 (it doesn't support 8-bit)
    "clip_bits": 32,
    # device to run projector on
    "projector_device": None,
    # projector bits, either 32 or 16
    "projector_bits": 32
}

# Initialize Annoy index
annoy_index = AnnoyIndex(VECTOR_DIM, 'angular')

if os.path.exists(DATABASE_PATH):
    # Load existing index into a temporary index
    temp_index = AnnoyIndex(VECTOR_DIM, 'angular')
    temp_index.load(DATABASE_PATH)

    # Clone items from temporary index to the new index
    for i in range(temp_index.get_n_items()):
        annoy_index.add_item(i, temp_index.get_item_vector(i))
    
    temp_index.unload()
else:
    # Ensure the directory for the index file exists
    index_dir = os.path.dirname(DATABASE_PATH)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

def save_on_exit():
    """
    Save the Annoy index on program exit.
    """
    annoy_index.build(10)
    annoy_index.save(DATABASE_PATH)
    print("Annoy index saved on exit.")

# Register save_on_exit function to be called when the program is closing
atexit.register(save_on_exit)

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

def extract_summary_from_response(response):
    # Find the summary text in the response using a regex pattern
    summary_pattern = r"Summary:(.+)"
    summary_match = re.search(summary_pattern, response, flags=re.MULTILINE)

    if summary_match:
        summary = summary_match.group(1)
    else:
        summary = response

    return summary

async def generate_summary(text_to_summarize, state):
    
    # Get turn templates
    user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, True)

    rows = []
    # Adding the user message
    if len(text_to_summarize) > 0:
        rows.append(
            replace_all(
                user_turn, 
                {
                    '<|user-message|>': "Return only a summary of the CONTENT:\n<|CONTENT|>\n" + text_to_summarize.strip() + "\n<|END_CONTENT|>\n",
                    '<|round|>': "0"
                }
            )
        )

    # Adding the Character prefix
    rows.append(apply_extensions("bot_prefix", bot_turn_stripped.rstrip(' ')))

    
    # Format the prompt for summarization
    summary_prompt = ''.join(rows)
    print(f"summary_prompt:\n\n{summary_prompt}\n\n")

    # Send the prompt to the LLM and retrieve the response
    summary_response_generator = generate_reply(summary_prompt, state)

    # Get the full response from the generator
    response = ""
    for response_part in summary_response_generator:
        response = response_part
    summary_response = response

    # Process the response to extract the summary
    print(f"summary_response:\n\n{summary_response}\n\n")
    summary = extract_summary_from_response(summary_response)
    if len(summary) == 0:
        return text_to_summarize
    else:
        return summary


def _get_device(setting_name):
    if params[setting_name] is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(params[setting_name])

projector_device = _get_device('projector_device')

def generate_embeddings(text):
    input_ids = shared.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(projector_device)  # Move input_ids to the model's device

    with torch.no_grad():
        input_embeds = shared.model.model.embed_tokens(input_ids)

    input_embeds = input_embeds.mean(dim=1).squeeze(0)  # Remove the extra dimension
    return input_embeds.cpu().numpy().flatten()  # Convert to NumPy array and flatten

async def store_conversation(input_str, output_str, state):
    
    # Get turn templates
    user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, state['mode'] == 'instruct')

    rows = []
    
    # Adding the user history message
    i = len(shared.history['internal']) - 1
    if i >= 0:
        string = shared.history['internal'][i][0]
        rows.insert(0, replace_all(user_turn, {'<|user-message|>': string.strip(), '<|round|>': str(i)}))
    # Adding the Character message
    rows.append(bot_turn.replace('<|bot-message|>', output_str.strip()))
    # Adding the user message
    if len(input_str) > 0:
        rows.append(replace_all(user_turn, {'<|user-message|>': input_str.strip(), '<|round|>': str(len(shared.history["internal"]))}))


    memory = ''.join(rows)
    embedding = generate_embeddings(memory)

    annoy_index.add_item(i, embedding)

async def async_store_conversation(input_str, output_str, state):
    await asyncio.sleep(0)
    await store_conversation(input_str, output_str, state)

def retrieve_related_memories(input_str, history_rows, num_related_memories=10):
    #Clone and build annoy index
    annoy_index_clone = AnnoyIndex(VECTOR_DIM, 'angular')
    # Clone items from temporary index to the new index
    for i in range(annoy_index.get_n_items()):
        annoy_index_clone.add_item(i, annoy_index.get_item_vector(i))

    annoy_index_clone.build(10)
    annoy_index_clone.save(DATABASE_PATH)

    input_embedding = generate_embeddings(input_str)
    related_indices = annoy_index_clone.get_nns_by_vector(input_embedding, num_related_memories)
    annoy_index_clone.unload()
    print(f"Number of items in the Annoy index: {annoy_index.get_n_items()}")
    print(f"Input embedding: {input_embedding}")
    print(f"RELATED_INDICES: {related_indices}")  # Print the indices
    
    related_memories = [shared.history['internal'][max(0, index - 2):index] for index in related_indices]
    
    found_mem_str = '\n\n'.join(['\n'.join(t) for sublist in related_memories for t in sublist])
    print(f"FOUND_RELATED_MEMORIES:\n\n{found_mem_str}\n\n")

    # Filter out memories that are already present in the history added to the prompt
    non_duplicate_memories = [
        memory for memory in related_memories
        if ''.join(''.join(mem) for mem in memory) not in ''.join(history_rows)
    ]


    non_dup_mem_str = '\n\n'.join(['\n'.join(t) for sublist in non_duplicate_memories for t in sublist])
    print(f"NON_DUP_MEMS:\n\n{non_dup_mem_str}\n\n")
    
    return non_duplicate_memories


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def custom_generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    is_instruct = state['mode'] == 'instruct'
    rows = [state['context'] if is_instruct else f"{state['context'].strip()}\n"]
    min_rows = 3

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]

    max_length = min(get_max_prompt_length(state), chat_prompt_size)
    max_memory_length = floor(max_length * 0.3) - len(encode("Memories:\n\n\nChat:\n")[0])
    print(f"max_memory_length: {str(max_memory_length)}")
    # Get turn templates
    user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, is_instruct)

    # Building the prompt
    memories_header = "\nMemories:\n"
    chat_header = "\nChat:\n"
    mem_head_len = len(encode(memories_header)[0])
    chat_head_len = len(encode(chat_header)[0])
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

        i -= 1

    # Adding related memories to the prompt
    rows.append(memories_header)
    related_memories = retrieve_related_memories(user_input, history_rows)
    print(f"Related memories: {related_memories}")

    memory_len = 0
    memory_index = 0
    memory_rows = [];

    while memory_index < len(related_memories):
        memory = related_memories[memory_index]
        user_memory, ai_memory = memory[0][0], memory[1][1]
        proposed_user_turn = replace_all(user_turn, {'<|user-message|>': user_memory.strip(), '<|round|>': str(memory_index)})
        proposed_bot_turn = bot_turn.replace('<|bot-message|>', ai_memory.strip())
        
        new_memory_len = memory_len + len(encode(proposed_user_turn)[0]) + len(encode(proposed_bot_turn)[0])
        
        if new_memory_len <= max_memory_length:
            memory_len = new_memory_len
            memory_rows.extend([proposed_user_turn, proposed_bot_turn])
        else:
            break

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
    print(f"custom_generated_prompt:\n\n{prompt}\n\n")
    print(f"prompt_len:{len(encode(prompt)[0])}\nmax_length:{max_length}\nmax_memory_length:{max_memory_length}\nmax_history_length:{max_history_length}\nmax_content_length:{max_history_length+max_memory_length}\ntotal_content_length:{len(encode(rows[0])[0]) + max_history_length + max_memory_length}")
    last_response = modules.shared.history['internal'][-1][1] if len(modules.shared.history['internal']) > 0 else None
    if last_response:
        run_async(async_store_conversation, user_input, last_response, state)
    if also_return_rows:
        return prompt, rows
    else:
        return prompt
