from typing import List
from bertopic import BERTopic
from top2vec import Top2Vec
from topic_modeling.entity.config_entity import EmbeddingModelConfig
from topic_modeling.utils.helpers import get_num_workers

class EmbeddingModel:
    def __init__(self, config: EmbeddingModelConfig, model_type: str, num_topics: int, top_n: int):
        self.config = config
        self.model_type = model_type.upper()
        self.num_topics = num_topics
        self.top_n = top_n
        self.num_workers = get_num_workers()
        self.model = None

    def fit(self, texts: List[str]):
        if self.model_type == 'BERTOPIC':
            self.model = BERTopic(nr_topics=self.num_topics,
                                min_topic_size=self.config.min_topic_size, 
                                top_n_words=self.config.top_n_words,
                                calculate_probabilities=self.config.calculate_probabilities,
                                language=self.config.language,
                                n_gram_range=self.config.n_gram_range,
                                low_memory=self.config.low_memory,)
            self.model.fit(texts)
        elif self.model_type == 'TOP2VEC':
            # Top2Vec learns the number of topics automatically, then we reduce it
            self.model = Top2Vec(texts, speed=self.config.speed, workers=min(self.num_workers, self.config.workers))
            if self.num_topics < self.model.get_num_topics():
                self.model.hierarchical_topic_reduction(num_topics=self.num_topics)

        else:
            raise ValueError(f"Model type {self.model_type} not recognized for EmbeddingModel.")

    def get_topics(self) -> List[List[str]]:
        if self.model_type == 'BERTOPIC':
            # Extract topics from BERTopic internal representation
            all_topics = self.model.get_topics()
            # -1 is the outlier topic in BERTopic
            return [[word for word, _ in all_topics[i][:self.top_n]]
                    for i in sorted(all_topics.keys()) if i != -1]

        elif self.model_type == 'TOP2VEC':
            topic_words, word_scores, topic_nums = self.model.get_topics(num_topics=self.num_topics)
            return [list(w[:self.top_n]) for w in topic_words]