# ./text_preprocessor.py
import spacy
from extensions.annoy_ltm.helpers import *

class TextPreprocessor:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])

    def preprocess_and_extract_named_entities(self, text):
            # Named Entity Recognition
            doc = self.nlp(text)
            named_entities = [ent.text for ent in doc.ents]

            return named_entities

    def preprocess_and_extract_keywords(self, text):
        # Tokenization, lowercasing, and stopword removal
        tokens = [token.text.lower() for token in self.nlp(text) if not token.is_stop]

        # Lemmatization
        lemmatized_tokens = [token.lemma_ for token in self.nlp(" ".join(tokens))]

        keywords = lemmatized_tokens

        return keywords

    def trim_and_preprocess_text(self, text, state):
        text_to_process = remove_username_and_timestamp(text, state)
        keywords = self.preprocess_and_extract_keywords(text_to_process)
        named_entities = self.preprocess_and_extract_named_entities(text_to_process)

        return keywords, named_entities