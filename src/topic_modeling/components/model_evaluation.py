import os
import json
import pandas as pd
from typing import List, Dict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, vocab: List[str], cleaned_docs: List[str]):
        """
        Args:
            vocab: List of vocabulary words.
            cleaned_docs: List of cleaned document strings (from transformation stage).
        """
        self.config = config
        self.vocab = vocab
        # Coherence requires tokenized words: list of lists of strings
        self.tokenized_corpus = [doc.split() for doc in cleaned_docs]
        # FIX: Build dictionary from the model's vocabulary (self.vocab), not just test_clean_texts
        # This ensures all words appearing in `topics` are present in the dictionary.
        self.dictionary = Dictionary([self.vocab])

    def calculate_metrics(self, topics: List[List[str]], model_name: str) -> Dict:
        """Computes Coherence and Diversity for a given set of topics."""
        logger.info(f"Calculating metrics for {model_name}...")

        # 1. Coherence (Cv)
        cm = CoherenceModel(
            topics=topics,
            texts=self.tokenized_corpus,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        cv_score = cm.get_coherence()

        # 2. Topic Diversity
        all_words = [word for topic in topics for word in topic]
        diversity = len(set(all_words)) / len(all_words) if all_words else 0

        return {
            "model": model_name,
            "coherence_cv": round(cv_score, 4),
            "diversity": round(diversity, 4)
        }

    def save_results(self, metrics: Dict, topics: List[List[str]]):
        """Saves metrics to CSV and topic words to JSON."""
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Save Metrics to CSV
        df = pd.DataFrame([metrics])
        if not self.config.report_csv.exists():
            df.to_csv(self.config.report_csv, index=False)
        else:
            df.to_csv(self.config.report_csv, mode='a', header=False, index=False)

        # Save Topics to JSON (useful for dashboarding later)
        topic_data = {metrics['model']: topics}
        if self.config.top_words_json.exists():
            with open(self.config.top_words_json, 'r') as f:
                existing_data = json.load(f)
            existing_data.update(topic_data)
            topic_data = existing_data

        with open(self.config.top_words_json, 'w') as f:
            json.dump(topic_data, f, indent=4)

        logger.info(f"Evaluation artifacts saved in {self.config.root_dir}")