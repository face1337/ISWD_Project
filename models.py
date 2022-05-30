from tensorflow.keras import Model, layers
from typing import Dict

import tensorflow as tf


class RankingModel(Model):
  def __init__(self, user_vocab, movie_vocab):
    super().__init__()

    # Przygotowanie użytkowników oraz słownika z filmami
    self.user_vocab = user_vocab
    self.movie_vocab = movie_vocab
    self.user_embed = layers.Embedding(user_vocab.vocabulary_size(),
                                                64)
    self.movie_embed = layers.Embedding(movie_vocab.vocabulary_size(),
                                                 64)

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    # Określenie jak filmy są dobierane, np. na podstawie tytułu:

    embeddings_user= self.user_embed(self.user_vocab(features["user_id"]))
    embeddings_movie = self.movie_embed(
        self.movie_vocab(features["movie_title"]))

    return tf.reduce_sum(embeddings_user * embeddings_movie, axis=2)
