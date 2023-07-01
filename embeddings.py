# ./embeddings.py

from modules import shared
from extensions.annoy_ltm.helpers import get_device

import torch

def generate_embeddings(text, logger):
    """
    Generates embeddings for a given text.
    
    Parameters:
    text (str): The input text to generate embeddings for.
    logger (logging.Logger): A logger to log the process.

    Returns:
    np.ndarray: The generated embeddings.
    """
        
    input_ids = shared.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.long() # ensure the values are not floats

    with torch.no_grad():
        if hasattr(shared.model, 'ex_model'):
            input_ids = input_ids.to(torch.device("cpu"))  # Move input_ids to the model's device
            input_embeds = shared.model.ex_model.embed_tokens(input_ids)
        elif hasattr(shared.model.model, 'embed_tokens'):
            input_ids = input_ids.to(get_device())  # Move input_ids to the model's device
            input_embeds = shared.model.model.embed_tokens(input_ids)
        elif hasattr(shared.model.model, 'get_input_embeddings'):
            input_ids = input_ids.to(get_device())  # Move input_ids to the model's device
            input_embeds = shared.model.model.get_input_embeddings()(input_ids)
        elif hasattr(shared.model.model, 'model'): # Reported in issue #17
            input_ids = input_ids.to(get_device())  # Move input_ids to the model's device
            if hasattr(shared.model.model.model, 'embed_tokens'):
                input_embeds = shared.model.model.model.embed_tokens(input_ids)
        else:
            raise AttributeError("The model doesn't have an 'embed_tokens' or 'get_input_embeddings' method.")

    input_embeds = input_embeds.mean(dim=1).squeeze(0)  # Remove the extra dimension
    result = input_embeds.cpu().numpy().flatten()  # Convert to NumPy array and flatten
    logger(f"generating embeddings for text: {text}\n{result}", 5)
    return result

class Embeddings:
    def __init__(self) -> None:
        self.embeddings = {}

    def add_embedding(self, unique_index, embeddings):
        self.embeddings[unique_index] = embeddings
    
    def export_embeddings(self):
        return self.embeddings
    
    def import_embeddings(self, embeddings):
        self.embeddings = embeddings