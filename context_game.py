import random
import time

import numpy as np

from load import load_cached_data

class ContextGame:
    def __init__(self, vocab: list[str], embeddings: np.ndarray, word_to_idx: dict[str, int]):
        self.vocab = vocab
        self.embeddings = embeddings
        self.word_to_idx = word_to_idx
        self.secret_idx = None
        self.rank = None

    def select_secret_word(self) -> None:
        """Selects the target word to be guessed"""
        self.secret_idx = random.randint(0, len(self.vocab) - 1)

        # Initialize embedding of secret word
        secret_vector = self.embeddings[self.secret_idx]

        # Compute similarity of of secret vector with all other words in the vocabulary
        similarities = self.embeddings @ secret_vector
        order = np.argsort(-similarities)
        self.rank = {i: rank for rank, i in enumerate(order)}

    def guess(self, word: str) -> int:
        """
            Evaluate a guessed word and return its rank (1 = best / secret word).
            Raises ValueError if the word is not in the vocabulary or if the game
            has not been initialized with a secret word.
        """
        if self.rank is None or self.secret_idx is None:
            raise ValueError("Secret word not selected. Call select_secret_word() first.")
        word = word.strip().lower()

        if word not in self.word_to_idx:
            raise ValueError(f"Unknown word: {word!r} (not in vocabulary)")
        
        # Retrieve index of guessed word
        idx = self.word_to_idx[word]
        # Get rank of guessed word
        rank0 = self.rank[idx]
        # Add to make ranking 1-based for user
        return rank0 + 1
    
    def is_correct(self, rank: int) -> bool:
        """Check if guessed word is correct"""
        if isinstance(self.secret_idx, int):
            return rank == 1
        return False
    
    def reveal_secret(self) -> str:
        """Reveal secret word. Raises an error if secret word is not selected"""
        if self.secret_idx is None:
            raise ValueError("Secret word has not been selected yet.")
        return self.vocab[self.secret_idx]


def run_game() -> None:
    print("Loading Vocabulary and Embedding Matrix...")
    vocab, embeddings, word_to_idx = load_cached_data()
    game = ContextGame(vocab, embeddings, word_to_idx)
    game.select_secret_word()

    print("Game started! Try to guess the secret word.")
    # For debugging:
    # print("DEBUG – secret word is:", game.reveal_secret())

    num_guesses = 0
    game_start = time.time()

    while True:
        user_guess = input("Please enter your guess: ").strip()

        if not user_guess.isalpha():
            print("Please enter a valid word (alphabetic strings only).")
            continue

        try:
            guess_rank = game.guess(user_guess)
        except ValueError as e:
            # e.g., "Unknown word: 'foo' (not in vocabulary)"
            print(e)
            continue

        num_guesses += 1

        if game.is_correct(guess_rank):
            elapsed = time.time() - game_start
            print(f"Congratulations! You guessed the correct word in {num_guesses} guesses.")
            print(f"⏱ Total time: {elapsed / 60:.2f} minutes")
            print(f"The secret word was: {game.reveal_secret()}")
            break

        print(f"Rank of '{user_guess.lower()}': {guess_rank}")


if __name__ == "__main__":
    run_game()
