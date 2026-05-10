"""
data.py
Data loading, sampling, train/test split, and tokenizer encoding.

Usage:
    python data.py
"""

import gzip
import json
import pickle
import random
import requests

from transformers import DistilBertTokenizerFast

import utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 128
HEAD = 10000        # maximum reviews to stream per genre
SAMPLE_SIZE = 200   # reduced for CPU training
TRAIN_PER_GENRE = 160
TEST_PER_GENRE = 40
RANDOM_SEED = 42
PICKLE_PATH = "genre_reviews_dict.pickle"

GENRE_URL_DICT = {
    "poetry": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_poetry.json.gz"
    ),
    "comics_graphic": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_comics_graphic.json.gz"
    ),
    "fantasy_paranormal": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_fantasy_paranormal.json.gz"
    ),
    "history_biography": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_history_biography.json.gz"
    ),
    "mystery_thriller_crime": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_mystery_thriller_crime.json.gz"
    ),
    "romance": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_romance.json.gz"
    ),
    "young_adult": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
        "goodreads_reviews_young_adult.json.gz"
    ),
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_reviews(url: str, head: int = HEAD, sample_size: int = SAMPLE_SIZE) -> list[str]:
    """Stream reviews from a gzipped JSON-lines URL and return a random sample."""
    reviews = []
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(response.raw, "rt", encoding="utf-8") as fh:
        for count, line in enumerate(fh):
            record = json.loads(line)
            reviews.append(record["review_text"])
            if head is not None and count + 1 >= head:
                break
    random.seed(RANDOM_SEED)
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(
    genre_url_dict: dict = GENRE_URL_DICT,
    pickle_path: str = PICKLE_PATH,
) -> dict:
    """
    Load reviews for every genre.
    If a pickle file already exists at pickle_path, load from it instead of
    re-downloading.
    """
    try:
        with open(pickle_path, "rb") as fh:
            print(f"Loaded genre reviews from cache: {pickle_path}")
            return pickle.load(fh)
    except FileNotFoundError:
        pass

    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url)

    with open(pickle_path, "wb") as fh:
        pickle.dump(genre_reviews_dict, fh)
    print(f"Saved genre reviews to cache: {pickle_path}")
    return genre_reviews_dict


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_data(
    genre_reviews_dict: dict,
    train_per_genre: int = TRAIN_PER_GENRE,
    test_per_genre: int = TEST_PER_GENRE,
) -> tuple[list, list, list, list]:
    """Return (train_texts, train_labels, test_texts, test_labels)."""
    random.seed(RANDOM_SEED)
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    n = train_per_genre + test_per_genre
    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(n, len(reviews)))
        for text in sampled[:train_per_genre]:
            train_texts.append(text)
            train_labels.append(genre)
        for text in sampled[train_per_genre:n]:
            test_texts.append(text)
            test_labels.append(genre)
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode_data(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
):
    """
    Tokenize texts and encode string labels to integers.
    Returns (train_dataset, test_dataset) as GoodreadsDataset objects.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_length
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=max_length
    )

    label2id, id2label = utils.build_label_maps(train_labels)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    train_dataset = utils.GoodreadsDataset(train_encodings, train_labels_encoded)
    test_dataset = utils.GoodreadsDataset(test_encodings, test_labels_encoded)

    return tokenizer, train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    genre_reviews_dict = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    print(
        f"Split sizes: train={len(train_texts)}, test={len(test_texts)}"
    )
    tokenizer, train_dataset, test_dataset = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )
    print(
        f"Datasets created: train={len(train_dataset)}, test={len(test_dataset)}"
    )
    print(f"Label map: {utils.label2id}")


if __name__ == "__main__":
    main()
