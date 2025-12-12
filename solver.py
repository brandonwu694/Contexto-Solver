import random
from abc import ABC, abstractmethod

import numpy as np

from load import load_cached_data
from game import ContextGame


STARTER_WORDS = [
            "thing", "object", "person", "world", "life",
            "work", "system", "group", "water", "animal",
            "city", "country", "family", "machine", "food",
        ]


class Solver(ABC):
    """Abstract base class for all Contexto solver algorithms."""
    def __init__(self, vocab: list[str], embeddings: np.ndarray, word_to_idx: dict[str, int]):
        self.vocab: list[str] = vocab
        self.embeddings: np.ndarray = embeddings
        self.word_to_idx: dict[str, int] = word_to_idx
        self.start_word: str | None = None
        self.starter_words: list[str] = [word for word in STARTER_WORDS if word in self.vocab]
        self.guessed: list[tuple[str, int]] = []
    
    def select_start_word(self) -> str:
        """Select the first guess in the puzzle"""
        if not self.starter_words:
            word = random.choice(self.vocab)
        else:
            word = random.choice(self.starter_words)
        self.start_word = word
        return word
    
    def reset(self) -> None:
        """Reset solver state"""
        self.guessed.clear()
        self.start_word = None

    @abstractmethod
    def solve(self, game: ContextGame) -> int:
        """Implementation of algorithm that solves Contexto puzzle"""
        pass
