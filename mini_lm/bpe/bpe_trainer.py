import os
from typing import BinaryIO, Optional
from collections import defaultdict
import regex as re
import heapq
import multiprocessing
from .word import Word
from .bpe_model import BpeModel
from mini_lm.config.logging_config import get_logger


class Merge:
    """A helper class to store information about a potential merge."""

    def __init__(
        self,
        pair: tuple[int, int],
        count: int,
        pos: set[int],
        pair_bytes: tuple[bytes, bytes] = None,
    ):
        self.pair = pair
        self.count = count
        self.pos = pos
        self.pair_bytes = pair_bytes

    def __lt__(self, other: "Merge") -> bool:
        """Comparison for max heap behavior. Higher counts are prioritized."""
        if self.count != other.count:
            return self.count > other.count
        # Tie-breaking: choose lexicographically greater pair of bytes for determinism.
        return self.pair_bytes > other.pair_bytes

    def __repr__(self) -> str:
        return f"Merge(count={self.count}, pair={self.pair})"


class BpeTrainer:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer from a text corpus.

    This class handles the pre-tokenization, pair counting, and merging steps
    of the BPE training process. It supports parallel pre-tokenization for
    improved performance on large files.
    """

    def __init__(self) -> None:
        """Initializes the BpeTrainer with the GPT-2 pre-tokenization pattern."""
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.logger = get_logger(__name__)

    def do_train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_workers: Optional[int] = None,
    ) -> BpeModel:
        """
        Trains a BPE model from a text file.

        Args:
            input_path (str): The path to the training text file.
            vocab_size (int): The desired final vocabulary size.
            special_tokens (list[str]): A list of special tokens to be included in the vocabulary.
            num_workers (Optional[int]): The number of parallel workers for pre-tokenization.
                                         Defaults to the number of CPU cores.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                A tuple containing the vocabulary (ID to byte mapping) and the list of merges.
        """
        word_counts: dict[bytes, int] = {}

        # 0. Pre-tokenize the input file to get initial word counts.
        self.logger.info("Starting pre-tokenization", input_path=input_path)
        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, special_tokens)
            num_procs = num_workers or multiprocessing.cpu_count()
            self.logger.info("Using parallel pre-tokenization", num_workers=num_procs, num_chunks=len(boundaries))
            word_counts = self._parallel_pretokenization_safe(
                input_path, boundaries, num_procs
            )
        self.logger.info("Pre-tokenization complete", unique_words=len(word_counts))

        # Initialize vocabulary and merges.
        word_to_id: dict[bytes, int] = {}
        id_to_word: dict[int, bytes] = {}

        # 1. Add special tokens and initial alphabet to the vocabulary.
        self._add_special_tokens(word_to_id, id_to_word, special_tokens)
        self._compute_alphabet(word_to_id, id_to_word)
        self.logger.info("Initial vocabulary created", vocab_size=len(word_to_id))

        # 2. Tokenize words into sequences of base characters.
        words, counts = self._tokenize_words(word_counts, word_to_id)
        self.logger.info("Words tokenized", num_words=len(words))

        # 3. Count initial pairs of symbols.
        pair_counts, where_to_update = self._count_pairs(words, counts)
        self.logger.info("Initial pairs counted", unique_pairs=len(pair_counts))

        # 4. Initialize a priority queue with the initial pairs.
        pq = self._initialize_priority_queue(pair_counts, where_to_update, id_to_word)
        self.logger.info("Priority queue initialized", queue_size=len(pq))

        # 5. Perform merges until the vocabulary size is reached.
        merges = self._perform_merges(
            word_to_id,
            id_to_word,
            vocab_size,
            pair_counts,
            words,
            pq,
            counts,
            where_to_update,
        )
        self.logger.info("Training complete", total_merges=len(merges), final_vocab_size=len(word_to_id))

        return BpeModel(id_to_word, merges, special_tokens)

    def _compute_alphabet(self, w2id: dict[bytes, int], id2w: dict[int, bytes]):
        """Populates the vocabulary with all possible single-byte tokens."""
        next_id = len(id2w)
        for i in range(256):
            token_bytes = bytes([i])
            if token_bytes not in w2id:
                w2id[token_bytes] = next_id
                id2w[next_id] = token_bytes
                next_id += 1

    def _add_special_tokens(
        self, w2id: dict[bytes, int], id2w: dict[int, bytes], special_tokens: list[str]
    ):
        """Adds special tokens to the vocabulary."""
        next_id = len(id2w)
        for token_str in special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes not in w2id:
                w2id[token_bytes] = next_id
                id2w[next_id] = token_bytes
                next_id += 1

    def _tokenize_words(
        self, wc: dict[bytes, int], w2id: dict[bytes, int]
    ) -> tuple[list[Word], list[int]]:
        """
        Converts words (byte strings) into sequences of initial token IDs.

        Args:
            wc (dict[bytes, int]): A dictionary mapping words to their frequencies.
            w2id (dict[bytes, int]): A dictionary mapping token bytes to their IDs.

        Returns:
            tuple[list[Word], list[int]]: A tuple containing a list of Word objects
                                          and a corresponding list of their frequencies.
        """
        words = []
        counts = []
        byte_to_id = {i: w2id[bytes([i])] for i in range(256)}

        for word_bytes, count in wc.items():
            current_word = Word()
            for byte_val in word_bytes:
                current_word.add(byte_to_id[byte_val], 1)
            words.append(current_word)
            counts.append(count)

        return words, counts

    def _count_pairs(
        self, words: list[Word], counts: list[int]
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
        """Counts all adjacent pairs of symbols in the words.

        Args:
            words (list[Word]): the total words from the input source.
            counts (list[int]): the frequency of the word in words

        Returns:
            tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
                A tuple containing the pair frequency map and a map from pairs to the set of word indices
                where the pair occurs.
        """
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where_to_update: dict[tuple[int, int], set[int]] = defaultdict(set)

        for i, word in enumerate(words):
            for j in range(len(word.symbols) - 1):
                pair = (word.symbols[j].c, word.symbols[j + 1].c)
                pair_counts[pair] += counts[i]
                where_to_update[pair].add(i)

        return pair_counts, where_to_update

    def _perform_merges(
        self,
        w2id: dict[bytes, int],
        id2w: dict[int, bytes],
        vocab_size: int,
        pair_counts: dict[tuple[int, int], int],
        words: list[Word],
        pq: list[Merge],
        counts: list[int],
        where_to_update: dict[tuple[int, int], set[int]],
    ) -> list[tuple[bytes, bytes]]:
        """
        Performs the BPE merge operations until the desired vocabulary size is reached.

        This method iteratively finds the most frequent pair, merges it, updates the
        pair counts, and adds new pairs to the priority queue.
        """
        merges = []
        iteration = 0
        while len(w2id) < vocab_size and pq:
            if iteration % 1000 == 0:
                self.logger.info("Merge progress", iteration=iteration, vocab_size=len(w2id), target_vocab_size=vocab_size)
            iteration += 1

            top = heapq.heappop(pq)

            # If the count in the priority queue is stale, update it and re-queue.
            if top.count != pair_counts.get(top.pair, 0):
                top.count = pair_counts[top.pair]
                heapq.heappush(pq, top)
                continue

            if top.count < 1:
                break

            assert top.pair[0] in id2w and top.pair[1] in id2w
            part_a, part_b = id2w[top.pair[0]], id2w[top.pair[1]]
            new_token = part_a + part_b

            # Check if we can add the new token to vocabulary
            if new_token not in w2id:
                if len(w2id) >= vocab_size:
                    # Vocabulary is full, stop merging
                    self.logger.info(
                        "Vocabulary size limit reached, stopping merges",
                        merge_index=len(merges),
                        vocab_size=len(w2id),
                        vocab_limit=vocab_size,
                    )
                    break
                
                # Add new token to vocabulary
                new_id = len(w2id)
                w2id[new_token] = new_id
                id2w[new_id] = new_token
                self.logger.debug(
                    "Added new token to vocabulary",
                    merge_index=len(merges),
                    part_a=repr(part_a),
                    part_b=repr(part_b),
                    new_token=repr(new_token),
                    new_id=new_id,
                    vocab_size=len(w2id),
                )
            else:
                # Token already exists
                new_id = w2id[new_token]
                self.logger.debug(
                    "Merged token already exists in vocabulary",
                    merge_index=len(merges),
                    part_a=repr(part_a),
                    part_b=repr(part_b),
                    new_token=repr(new_token),
                    existing_id=new_id,
                )

            # Only add the merge if the token is in vocabulary
            merges.append((part_a, part_b))

            # Update pair counts and word structures after the merge.
            changes = []
            for wid in top.pos:
                word = words[wid]
                local_changes = word.merge(top.pair[0], top.pair[1], new_id)
                for change in local_changes:
                    changes.append((change, wid))

            new_pairs_to_queue = set()
            for (pair, count_delta), wid in changes:
                pair_counts[pair] = pair_counts.get(pair, 0) + count_delta * counts[wid]
                if count_delta > 0:
                    where_to_update[pair].add(wid)
                    new_pairs_to_queue.add(pair)

            for pair in new_pairs_to_queue:
                pair_bytes = (id2w[pair[0]], id2w[pair[1]])
                heapq.heappush(
                    pq,
                    Merge(pair, pair_counts[pair], where_to_update[pair], pair_bytes),
                )
        return merges

    def _initialize_priority_queue(
        self,
        pair_counts: dict[tuple[int, int], int],
        where_to_update: dict[tuple[int, int], set[int]],
        id_to_word: dict[int, bytes],
    ) -> list[Merge]:
        """Initializes a priority queue with the initial pair counts."""
        pq = []
        for pair, count in pair_counts.items():
            if count > 0:
                pair_bytes = (id_to_word[pair[0]], id_to_word[pair[1]])
                pq.append(Merge(pair, count, where_to_update[pair], pair_bytes))
        heapq.heapify(pq)
        return pq

    def _parallel_pretokenization(
        self,
        input_path: str,
        boundaries: list[tuple[int, int]],
        num_workers: Optional[int] = None,
    ) -> dict[bytes, int]:
        """
        Performs pre-tokenization in parallel using a pool of worker processes.

        Args:
            input_path (str): The path to the input text file.
            boundaries (list[tuple[int, int]]): A list of (start, end) byte offsets for each chunk.
            num_workers (Optional[int]): The number of worker processes to use.

        Returns:
            dict[bytes, int]: A dictionary mapping pre-tokens to their frequencies.
        """
        worker_args = [(input_path, start, end, self.pat) for start, end in boundaries]
        num_procs = num_workers or min(multiprocessing.cpu_count(), len(boundaries))

        word_counts = defaultdict(int)
        with multiprocessing.Pool(processes=num_procs) as pool:
            chunk_results = pool.imap_unordered(
                BpeTrainer._process_chunk_worker, worker_args
            )
            for chunk_counts in chunk_results:
                for token, freq in chunk_counts.items():
                    word_counts[token] += freq
        return word_counts

    @staticmethod
    def _process_chunk_worker(args: tuple[str, int, int, str]) -> dict[bytes, int]:
        """Worker function to process a single chunk in parallel."""
        file_path, start, end, pattern = args
        try:
            with open(file_path, "rb") as f:
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk = chunk_bytes.decode("utf-8", errors="ignore")
            trainer = BpeTrainer()
            trainer.pat = pattern
            return trainer._pre_tokenize_chunk(chunk)
        except Exception as e:
            # Get a logger instance for the static method
            logger = get_logger(__name__)
            logger.error("Error processing chunk", start=start, end=end, error=str(e))
            return {}

    def _parallel_pretokenization_safe(
        self,
        input_path: str,
        boundaries: list[tuple[int, int]],
        num_workers: Optional[int] = None,
    ) -> dict[bytes, int]:
        """A wrapper for parallel pre-tokenization that falls back to sequential processing on error."""
        try:
            return self._parallel_pretokenization(input_path, boundaries, num_workers)
        except Exception as e:
            self.logger.warning(
                "Parallel processing failed, falling back to sequential",
                error=str(e),
                error_type=type(e).__name__
            )
            with open(input_path, "rb") as f:
                return self._sequential_pretokenization(f, boundaries)

    def _sequential_pretokenization(
        self, file_handle: BinaryIO, boundaries: list[tuple[int, int]]
    ) -> dict[bytes, int]:
        """Performs pre-tokenization sequentially, chunk by chunk."""
        word_counts = defaultdict(int)
        for start, end in boundaries:
            file_handle.seek(start)
            chunk_bytes = file_handle.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            chunk_pretoken_table = self._pre_tokenize_chunk(chunk_text)
            for token, freq in chunk_pretoken_table.items():
                word_counts[token] += freq
        return word_counts

    def _pre_tokenize_chunk(self, chunk_text: str) -> dict[bytes, int]:
        """Finds all pre-tokens in a chunk of text and counts their frequencies."""
        chunk_pretoken_counts = defaultdict(int)
        for match in re.finditer(self.pat, chunk_text):
            token_bytes = match.group(0).encode("utf-8")
            chunk_pretoken_counts[token_bytes] += 1
        return chunk_pretoken_counts

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        special_tokens: list[str],
    ) -> list[tuple[int, int]]:
        """
        Finds the boundaries of text chunks in a file, separated by special tokens.
        """
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # Handle empty file
        if file_size == 0:
            return []

        if not special_tokens:
            return [(0, file_size)]

        # Sort by length (longest first) to handle overlapping tokens correctly.
        special_tokens_bytes = sorted(
            [token.encode("utf-8") for token in special_tokens], key=len, reverse=True
        )
        pattern = b"|".join(re.escape(token) for token in special_tokens_bytes)
        max_token_len = len(special_tokens_bytes[0])

        token_positions = []
        chunk_size = 8192
        position = 0

        while position < file_size:
            file.seek(position)
            buffer = file.read(chunk_size + max_token_len - 1)
            if not buffer:
                break

            for match in re.finditer(pattern, buffer):
                if match.start() < chunk_size:
                    absolute_pos = position + match.start()
                    token_len = len(match.group(0))
                    token_positions.append((absolute_pos, token_len))
            position += chunk_size

        token_positions = sorted(list(set(token_positions)))

        boundaries = []
        current_pos = 0
        for token_pos, token_len in token_positions:
            if token_pos > current_pos:
                boundaries.append((current_pos, token_pos))
            current_pos = token_pos + token_len

        if current_pos < file_size:
            boundaries.append((current_pos, file_size))

        if not boundaries and file_size > 0:
            return [(0, file_size)]

        return boundaries