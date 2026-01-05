import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import DataEDAConfig



class TopicEDA:
    """
    Automated EDA engine for text datasets.
    Generates statistics and visualizations to inform topic model hyperparameter tuning.
    """
    def __init__(self, config: DataEDAConfig):
        self.config  = config
        # Set plot style for professional appearance
        sns.set_theme(style="whitegrid", palette="muted")

    def _get_top_ngrams(self, corpus: pd.Series, n: int = 1, top_k: int = 20):
        """Helper to calculate word/phrase frequencies."""
        words = " ".join(corpus.astype(str)).split()
        if n == 1:
            it = words
        else:
            it = [" ".join(gram) for gram in zip(*[words[i:] for i in range(n)])]

        return pd.DataFrame(Counter(it).most_common(top_k), columns=['ngram', 'count'])

    def run_full_analysis(self, df: pd.DataFrame):
        """Executes the full suite of EDA tests."""
        logger.info(f"Starting EDA on column: {self.config.text_col}")

        # 1. Basic Stats
        df['char_len'] = df[self.config.text_col].apply(len)
        df['word_count'] = df[self.config.text_col].apply(lambda x: len(str(x).split()))
        stats = df[['char_len', 'word_count']].describe()

        stats_path = self.config.root_dir / "summary_statistics.csv"
        stats.to_csv(stats_path)
        logger.info(f"Basic stats saved to {stats_path}")

        # 2. Plot Document Length Distribution
        self._plot_length_dist(df)

        # 3. Plot Top Unigrams and Bigrams
        self._plot_ngrams(df[self.config.text_col])

        # 4. Generate WordCloud
        self._plot_wordcloud(df[self.config.text_col])

        # 5. Label Distribution (if available)
        if self.config.label_col in df.columns:
            self._plot_label_dist(df, self.config.label_col)
        else:
            logger.warning(f"Label column '{self.config.label_col}' not found. Skipping label distribution plot.")

    def _plot_length_dist(self, df: pd.DataFrame):
        """Plots distribution of document lengths."""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['word_count'], kde=True, bins=50, color="teal")
        plt.title("Document Word Count Distribution")
        plt.xlabel("Words per Document")
        plt.ylabel("Frequency")
        plt.savefig(self.config.root_dir / "word_count_distribution.png")
        plt.close()

    def _plot_ngrams(self, corpus: pd.Series):
        """Generates bar charts for most frequent terms."""

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for i, n in enumerate([1, 2]):
            ngram_df = self._get_top_ngrams(corpus, n=n, top_k=self.config.top_k_ngrams)
            sns.barplot(data=ngram_df, x='count', y='ngram', ax=axes[i], palette="viridis", hue='ngram', legend=False)
            axes[i].set_title(f"Top 20 {'Unigrams' if n==1 else 'Bigrams'}")

        plt.tight_layout()
        plt.savefig(self.config.root_dir / "ngram_analysis.png")
        plt.close()

    def _plot_wordcloud(self, corpus: pd.Series):
        """Visual summary of most prominent words."""

        text = " ".join(corpus.astype(str))
        wc = WordCloud(width=self.config.wordcloud_width, height=self.config.wordcloud_height, background_color="white",
                       max_words=200, colormap="Dark2").generate(text)

        plt.figure(figsize=(15, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title("Corpus Word Cloud", fontsize=20)
        plt.savefig(self.config.root_dir / "wordcloud.png")
        plt.close()

    def _plot_label_dist(self, df: pd.DataFrame, label_col: str):
        plt.figure(figsize=(12, 6))
        sns.countplot(
            y=self.config.label_col,
            data=df,
            order=df[self.config.label_col].value_counts().index,
            palette="magma",
            hue=self.config.label_col,
            legend=False
        )
        plt.title(f"Category Distribution {self.config.label_col}")
        plt.tight_layout()
        plt.savefig(self.config.root_dir / "label_distribution.png")
        plt.close()
