# ./annoy_manager.py
from math import floor
import time

from modules import shared
from annoy import AnnoyIndex
import queue
import copy
import threading

from extensions.annoy_ltm.helpers import *
from extensions.annoy_ltm.metadata import check_hashes, compute_hashes, load_metadata, save_metadata
from extensions.annoy_ltm.embeddings import generate_embeddings
from extensions.annoy_ltm.keyword_tally import KeywordTally
from extensions.annoy_ltm.turn_templates import apply_turn_templates_to_rows

class AnnoyManager:
    def __init__(self, text_preprocessor) -> None:
        self.results_queue = queue.Queue()
        self.text_preprocessor = text_preprocessor
        self.metadata = None
        self.annoy_index = None
        self.metadata_file = None
        self.annoy_index_file = None
        self.hidden_size = None
        self.loaded_history_last_index = 0
        # Create dictionary for annoy indices
        self.index_to_history_position = {}
        self.lock = threading.Lock()

    def _get_hidden_size(self, params, logger):
        if params['vector_dim_override'] != -1:
            return params['vector_dim_override']
        try:
            if hasattr(shared.model, 'ex_config'):
                return shared.model.ex_config.hidden_size
            if hasattr(shared.model, 'config'):
                return shared.model.config.hidden_size
            
            return shared.model.model.config.hidden_size
        except AttributeError:
            return len(generate_embeddings('generate a set of embeddings to determin size of result list', logger=logger))

    def save_files_to_disk(self, logger):
        with self.lock:
            try:
                logger(f"Cloning data before saveing...", 3)
                metadata_to_save = copy.deepcopy(self.metadata)
                annoy_index_to_save = AnnoyIndex(self.hidden_size, 'angular')
                copy_items(self.annoy_index, annoy_index_to_save, self.annoy_index.get_n_items(), logger)

                logger(f"Saving metadata...", 3)
                save_metadata(metadata_to_save, self.metadata_file)
                logger(f"Metadata saved.", 3)
                logger(f"Saving annoy_index...", 3)
                annoy_index_to_save.build(10)
                annoy_index_to_save.save(self.annoy_index_file)
                logger(f"annoy_index saved.", 3)
            except Exception as e:
                logger(f"An error occurred while saving files to disk:\n{e}", level=1)
        

    def generate_annoy_db(self, params, state, history, keyword_tally, logger):
        with self.lock:
            try:
                # Generate annoy database for LTM
                start_time = time.time()

                self.metadata_file = f"{params['annoy_output_dir']}{state['name2']}-annoy-metadata.json"
                self.annoy_index_file = f"{params['annoy_output_dir']}{state['name2']}-annoy_index.ann"

                if self.metadata == None:
                    logger(f"Loading metadata file...", 5)
                    self.metadata = load_metadata(self.metadata_file)
                    logger(f"Loaded metadata.", 5)
                    if self.metadata == None:
                        logger(f"failed to load character annoy metadata, generating from scratch...", 1)
                    else:
                        logger(f"loaded metadata file ({len(self.metadata['messages_hash'])})", 2)
                

                hidden_size = self._get_hidden_size(params, logger)
                if self.annoy_index == None or self.hidden_size != hidden_size:
                    self.hidden_size = hidden_size  
                    loaded_annoy_index = AnnoyIndex(self.hidden_size, 'angular')
                    self.annoy_index = AnnoyIndex(self.hidden_size, 'angular')
                    
                    if check_hashes(self.metadata, history, logger):
                        logger(f"Loading annoy database...", 5)
                        loaded_annoy_index.load(self.annoy_index_file)
                        logger(f"Loaded database.", 5)
                        loaded_history_items = loaded_annoy_index.get_n_items()
                        if loaded_history_items < 1:
                            logger(f"hashes check passed but no items found in annoy db. rebuilding annoy db...", 2)
                        else:
                            logger(f"hashes check passed, proceeding to load existing memory db...", 2)
                            keyword_tally.importKeywordTally(self.metadata['keyword_tally'])
                            self.index_to_history_position = {int(k): v for k, v in self.metadata['index_to_history_position'].items()}
                            self.loaded_history_last_index = self.index_to_history_position[loaded_history_items-1]
                            logger(f"loaded {self.loaded_history_last_index} items from existing memory db", 3)
                            copy_items(loaded_annoy_index, self.annoy_index, loaded_history_items, logger)
                            loaded_annoy_index.unload()
                    else:
                        logger(f"hashes check failed, either an existing message changed unexpectdly or the extension code has changed. Rebuilding annoy db...", 2)
                        keyword_tally = KeywordTally()
                        self.loaded_history_last_index = 0

                formated_history_rows = apply_turn_templates_to_rows(history['internal'][self.loaded_history_last_index:], state, logger=logger)
                logger(f"found {len(formated_history_rows)} rows of chat history to be added to memory db. adding items...", 3)
                unique_index = len(self.index_to_history_position)
                for i, row in enumerate(formated_history_rows):
                    for msg in row:
                        trimmed_msg = remove_username_and_timestamp(msg, state)
                        if trimmed_msg and len(trimmed_msg) > 0:
                            # Add the full message
                            logger(f"HISTORY_{i+1}_MSG: {msg}", 4)
                            embeddings = generate_embeddings(trimmed_msg, logger=logger)
                            self.annoy_index.add_item(unique_index, embeddings)
                            self.index_to_history_position[unique_index] = i+self.loaded_history_last_index
                            unique_index += 1
                        
                            # Add keywords and named entities
                            keywords, named_entities = self.text_preprocessor.trim_and_preprocess_text(msg, state)
                            keyword_tally.tally(keywords + named_entities) # Keep a tally of all keywords and named_entities
                            filtered_keywords = filter_keywords(keywords)
                            keyword_groups = generate_keyword_groups(filtered_keywords, params['keyword_grouping'])
                            logger(f"HISTORY_{i+1}_KEYWORDS: {','.join(filtered_keywords)}", 4)
                            for keyword in keyword_groups:
                                embeddings = generate_embeddings(keyword, logger=logger)
                                logger(f"storing keyword \"{keyword}\" with embeddings {embeddings}", 5)
                                self.annoy_index.add_item(unique_index, embeddings)
                                self.index_to_history_position[unique_index] = i+self.loaded_history_last_index
                                unique_index += 1

                            if len(named_entities) > 0:
                                named_entities = " ".join(named_entities)
                                embeddings = generate_embeddings(named_entities, logger=logger)
                                logger(f"storing named_entities \"{named_entities}\" with embeddings {embeddings}", 5)
                                self.annoy_index.add_item(unique_index, embeddings)
                                self.index_to_history_position[unique_index] = i+self.loaded_history_last_index
                                unique_index += 1

                self.loaded_history_last_index += len(formated_history_rows)
                
                # Save the annoy index and metadata
                code_hash, messages_hash = compute_hashes(history)
                self.metadata = {
                    'code_hash': code_hash,
                    'messages_hash': messages_hash,
                    'model_name': shared.model_name,
                    'index_to_history_position': self.index_to_history_position,
                    'keyword_tally': keyword_tally.exportKeywordTally()
                }
                
                
                # Put the result in the queue.
                return_index = AnnoyIndex(self.hidden_size, 'angular')
                copy_items(self.annoy_index, return_index, self.annoy_index.get_n_items(), logger=logger)
                return_index_to_history_position = copy.copy(self.index_to_history_position)

                return_index.build(10)

                end_time = time.time()
                logger(f"building annoy index took {end_time-start_time} seconds...", 1)

                self.results_queue.put((return_index_to_history_position, return_index, keyword_tally))
                return return_index_to_history_position, return_index, keyword_tally
            
            except Exception as e:
                logger(f"An error occurred while generating annoy database:\n{e}", level=1)

    def generate_and_save(self, params, state, history, keyword_tally, logger):
        self.generate_annoy_db(params, state, history, keyword_tally, logger)
        self.save_files_to_disk(logger)
