# ./embeddings.py

from modules import shared
from extensions.annoy_ltm.helpers import _get_device

import torch

def generate_embeddings(text, logger):
    input_ids = shared.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(_get_device())  # Move input_ids to the model's device
    input_ids = input_ids.long() # ensure the values are not floats

    with torch.no_grad():
        input_embeds = shared.model.model.embed_tokens(input_ids)

    input_embeds = input_embeds.mean(dim=1).squeeze(0)  # Remove the extra dimension
    result = input_embeds.cpu().numpy().flatten()  # Convert to NumPy array and flatten
    logger(f"generating embeddings for text: {text}\n{result}", 5)
    return result