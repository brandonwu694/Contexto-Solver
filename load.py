import time
import pickle
from typing import Tuple

import numpy as np


# File glove.6B.300d.txt contains the vocabulary term followed by its vector embedding on each line
def _load_vocabulary(path: str = "data/glove.6B.300d.txt") -> Tuple[list[str], list[list[float]]]:
    """Load the set of words/tokens and their vector representations"""
    vocab, embeddings = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 301:
                continue

            word = parts[0].lower()
            if not word.isalpha():
                continue

            vocab.append(word)
            embeddings.append([float(num) for num in parts[1:]])

    return vocab, embeddings


def _normalize_embeddings(embeddings: list[list[float]]) -> np.ndarray:
    """
    Convert list of embedding vectors into a normalized NumPy array.
    Each vector will have length (L2 norm) = 1.
    """
    # Compute vector length for each row in embeddings
    embeddings_np = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    # avoid division by zero just in case
    norms[norms == 0] = 1.0
    return embeddings_np / norms


def _build_embedding_lookup(vocab: list[str]) -> dict[str, int]:
    """Map each vocabulary to its location in the embedding matrix"""
    return {word: i for i, word in enumerate(vocab)}


def build_vocabulary() -> Tuple[list[str], np.ndarray, dict[str, int]]:
    """Load words, normalize vector embeddings, and build a map of words to their embeddings"""
    vocab, embeddings = _load_vocabulary()
    embeddings_normalized = _normalize_embeddings(embeddings)
    word_to_idx = _build_embedding_lookup(vocab)
    return vocab, embeddings_normalized, word_to_idx


def save_loaded_data(vocab: list[str], embeddings: np.ndarray, word_to_idx: dict[str, int]) -> None:
    """Save vocab, embedding matrix, and lookup map to disk for fast loading"""
    with open("data/vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")

    # Save embeddings matrix (NumPy)
    np.save("data/embeddings.npy", embeddings)

    with open("data/word_to_idx.pkl", "wb") as f:
        pickle.dump(word_to_idx, f)


def load_cached_data() -> Tuple[list[str], np.ndarray, dict[str, int]]:
    """Load vocabulary, embeddings matrix, and lookup map from disk"""
    with open("data/vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    embeddings = np.load("data/embeddings.npy")

    with open("data/word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)

    return vocab, embeddings, word_to_idx


if __name__ == "__main__":
    print("=" * 60)
    print("Building Vocabulary and Embedding Matrix")
    print("=" * 60)

    start = time.time()

    try:
        vocab, embeddings, word_to_idx = build_vocabulary()
        save_loaded_data(vocab, embeddings, word_to_idx)

        elapsed = time.time() - start
        print("Build Complete!")
        print(f"Vocabulary size: {len(vocab):,}")
        print(f"Embedding dimensions: {embeddings.shape}")
        print(f"⏱Total build time: {elapsed:.3f} seconds")

    except Exception as e:
        print("Error: Failed to build vocabulary or embeddings.")
        print(f"Reason: {e}")
