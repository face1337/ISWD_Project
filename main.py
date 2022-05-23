import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_ranking as tfr
import pprint

from tensorflow.keras import optimizers, layers
from typing import Dict, Tuple
from models import RankingModel

# Dane ze zbioru filmów
ratings_data = tfds.load('movielens/100k-ratings', split="train")
features_data = tfds.load('movielens/100k-movies', split="train")

ratings_data = ratings_data.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

feature_data = features_data.map(lambda x: x["movie_title"])
users = ratings_data.map(lambda x: x["user_id"])

# Konwersja id użytkowników na indeksy integerów
user_ids_vocabulary = layers.experimental.preprocessing.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(users.batch(1000))

# Konwersja id filmów na indeksy integerów
movie_titles_vocabulary = layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(feature_data.batch(1000))

# Pogrupowanie po user_id
key_func = lambda x: user_ids_vocabulary(x["user_id"])
reduce_func = lambda key, dataset: dataset.batch(100)
train = ratings_data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

for x in train.take(1):
  for key, value in x.items():
    print(f"Tytuł filmu: {key}: {value.shape}")
    print(f"Przykładowi użytkownicy {key}: {value[:10].numpy()}")
    print()


def _features_and_labels(
    x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    labels = x.pop("user_rating")
    return x, labels

print(ratings_data)

train = train.map(_features_and_labels)

train = train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))

model = RankingModel(user_ids_vocabulary, movie_titles_vocabulary)

optimizer = optimizers.Adagrad(0.5)
loss = tfr.keras.losses.get(loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
eval_metrics = [tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
                tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)]
model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)

history = model.fit(train, epochs=9)

print('Wytrenowany model: \n')
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(history.history)

# Pobierz listę filmów z dostępnego zbioru (2000 rekordów)
for movie_titles in feature_data.batch(2000):
    break

selected_id = input("Podaj id użytkownika: ")

# Wygeneruj listę dla użytkownika o podanym id.
inputs = {
    "user_id":
        tf.expand_dims(tf.repeat("selected_id", repeats=movie_titles.shape[0]), axis=0),
    "movie_title":
        tf.expand_dims(movie_titles, axis=0)
}

# Get movie recommendations for user 42.
scores = model(inputs)
titles = tfr.utils.sort_by_scores(scores, [tf.expand_dims(movie_titles, axis=0)])[0]
print(f"Wygenerowana lista dla użytkownika nr: ,: {selected_id, titles[0, :10]}")
