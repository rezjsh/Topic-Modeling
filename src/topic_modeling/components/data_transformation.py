from typing import List
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm # auto detects if you are in a terminal or Jupyter notebook
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import DataTransformationConfig
import re

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tok2vec"])
        except OSError:
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tok2vec"])

        self.vectorizer = CountVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=tuple(self.config.ngram_range),
            stop_words='english'
        )

        self.vocab = None
        self.id2word = None


    def get_vocab(self) -> List[str]:
        """
        Returns the vocabulary as a list of strings.
        Ensures the vectorizer has been fitted first.
        """
        if self.vocab is not None:
            return self.vocab
        
        if self.vectorizer is not None and hasattr(self.vectorizer, 'get_feature_names_out'):
            self.vocab = self.vectorizer.get_feature_names_out().tolist()
            return self.vocab
        
        raise ValueError("Vocabulary not found. Ensure the vectorizer is fitted or vocab is loaded.")
    
    def get_id2word(self) -> dict:
        """Returns the mapping from word index to word."""
        if self.id2word is None:
            raise ValueError("id2word mapping is not set. Fit the vectorizer first.")
        return self.id2word

    def _basic_clean(self, text: str) -> str:
            """Removes URLs, Emails, and extra whitespace for both modes."""
            text = str(text).lower()
            text = re.sub(r'\S*@\S*\s?', '', text)  # Emails
            text = re.sub(r'http\S+', '', text)      # URLs
            text = re.sub(r'\s+', ' ', text).strip() # Extra whitespace
            return text
    
    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        Processes a list of documents.
        'aggressive': Lemmatized, no stops, no punct (Best for LDA/NTM).
        'contextual': Light cleaning, keeps structure (Best for BERT/Top2Vec).
        """
        logger.info(f"Preprocessing {len(texts)} documents in '{self.config.mode}' mode...")

        # Ensure input is list of strings
        texts = [str(t) for t in texts]
        cleaned_raw = [self._basic_clean(t) for t in texts]
        processed_docs = []

        for doc in tqdm(
            self.nlp.pipe(cleaned_raw, batch_size=self.config.batch_size, n_process=-1),
            total=len(cleaned_raw),
            desc=f"NLP Processing ({self.config.mode})"
        ):
            if self.config.mode == "aggressive":
                # Filter: no stop words, no punctuation, only alphabetic, length > 2
                tokens = [
                    token.lemma_ for token in doc
                    if not token.is_stop and not token.is_punct and token.is_alpha and len(token) > 2
                ]
            else:
                # Minimal filtering for embedding models
                tokens = [token.text for token in doc if not token.is_space]

            processed_docs.append(" ".join(tokens))
        
        return processed_docs

    def fit(self, train_texts: List[str]):
        """Fits the CountVectorizer on the training corpus."""
        logger.info("Fitting CountVectorizer on training data...")
        self.vectorizer.fit(train_texts)
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        self.id2word = {i: word for i, word in enumerate(self.vocab)}
        # Save vocabulary and id2word mapping as artifacts
        logger.info("Saving vocabulary and id2word mapping artifacts...")
        np.save(self.config.vocab_path, self.vocab)
        np.save(self.config.id2word_path, self.id2word)

        logger.info(f"Vocabulary size: {len(self.vocab)}")

    def transform_to_bow(self, texts: List[str]) -> np.ndarray:
        """Converts cleaned texts into a BoW matrix."""
        return self.vectorizer.transform(texts).toarray()

    def get_topic_words(self, topic_weights: np.ndarray, top_n: int = 10) -> List[List[str]]:
        """
        Helper to map topic-word weight indices back to actual words.
        Works for LDA, NMF, NTM, and ProdLDA results.
        """
        top_words = []
        for topic in topic_weights:
            top_indices = topic.argsort()[-top_n:][::-1]
            top_words.append([self.id2word[i] for i in top_indices])
        return top_words

    