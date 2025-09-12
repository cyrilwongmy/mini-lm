from collections.abc import Iterable, Iterator
import json
import os
import regex as re
from .word import Word
from mini_lm.config import get_logger


class Chunk:
    def __init__(self, start: int, end: int, special_token: str | None = None):
        """Represents a chunk of text or a special split token.

        Args:
            start (int): Start position of the chunk.
            end (int): End position of the chunk.
            special_token (str | None, optional): The special token if this chunk is a special token; otherwise None. Defaults to None.
        """
        self.start = start
        self.end = end
        self.special_token = special_token


class BpeModel:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initializes the BPE tokenizer.

        Args:
            vocab (dict[int, bytes]): Mapping from token IDs to token byte strings.
            merges (list[tuple[bytes, bytes]]): List of byte string pairs representing BPE merges.
            special_tokens (list[str] | None, optional): List of special tokens to preserve. Text will be split on these tokens. Defaults to None.
        """
        self.logger = get_logger(__name__)
        self.id_to_word = vocab
        self.word_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = merges

        # A map from a pair of token IDs to their merge rank and the resulting new token ID.
        # The merge rank is determined by the order of merges in the merges file.
        self.merge_map: dict[tuple[int, int], tuple[int, int]] = {}
        for i, (first, second) in enumerate(merges):
            self.merge_map[(self.word_to_id[first], self.word_to_id[second])] = (
                i,
                self.word_to_id[(first + second)],
            )

        self.special_tokens = special_tokens
        # This regex pattern is used to split text into pre-tokens. It's the same as the one used in GPT-2.
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # data field vocab to id_to_word
    @property
    def vocab(self) -> dict[int, bytes]:
        return self.id_to_word

    @classmethod
    def from_file(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "BpeModel":
        """Loads a BPE model from a vocabulary and merges file.

        Args:
            vocab_path (str): Path to the vocabulary file (a json mapping tokens to IDs).
            merges_path (str): Path to the merges file (each line is a pair of tokens to be merged).
            special_tokens (list[str] | None, optional): A list of special tokens. Defaults to None.

        Returns:
            BPE: A new BPE instance.
        """
        logger = get_logger(__name__)
        logger.info(
            "Loading BPE model from files",
            vocab_path=vocab_path,
            merges_path=merges_path,
        )

        with open(vocab_path) as vocab_f:
            vocab_data = json.load(vocab_f)

        vocab = {}
        for token_str, token_id in vocab_data.items():
            # Check if this is our special hex encoding for non-UTF-8 bytes
            if token_str.startswith("\\x") and len(token_str) > 2:
                try:
                    # Convert hex string back to bytes
                    token_bytes = bytes.fromhex(token_str[2:])
                except ValueError:
                    # If hex parsing fails, fall back to UTF-8 encoding
                    token_bytes = token_str.encode("utf-8")
            else:
                # Normal UTF-8 string
                token_bytes = token_str.encode("utf-8")
            vocab[token_id] = token_bytes
        logger.debug("Loaded vocabulary", vocab_size=len(vocab))

        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line:
                    # Split from the right (last space) to handle tokens that contain spaces
                    parts = cleaned_line.rsplit(" ", 1)
                    
                    # Handle hex-encoded tokens
                    if parts[0].startswith("\\x") and len(parts[0]) > 2:
                        try:
                            part1 = bytes.fromhex(parts[0][2:])
                        except ValueError:
                            part1 = parts[0].encode("utf-8")
                    else:
                        part1 = parts[0].encode("utf-8")
                    
                    if parts[1].startswith("\\x") and len(parts[1]) > 2:
                        try:
                            part2 = bytes.fromhex(parts[1][2:])
                        except ValueError:
                            part2 = parts[1].encode("utf-8")
                    else:
                        part2 = parts[1].encode("utf-8")
                    
                    merges.append((part1, part2))

        logger.info(
            "BPE model loaded",
            vocab_size=len(vocab),
            num_merges=len(merges),
            num_special_tokens=len(special_tokens) if special_tokens else 0,
        )
        return cls(vocab, merges, special_tokens)

    def save(self, folder: str, name: str):
        """Saves the BPE model to vocabulary and merges files.
        
        Args:
            folder (str): Directory path where files will be saved.
            name (str): Base name for the files (will create {name}_vocab.json and {name}_merges.txt).
        """
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        
        # Construct file paths
        vocab_path = os.path.join(folder, f"{name}_vocab.json")
        merges_path = os.path.join(folder, f"{name}_merges.txt")
        
        # Save vocab to json
        # Convert from {id: bytes} to {string: id} format to match from_file format
        vocab_data = {}
        for token_id, token_bytes in self.id_to_word.items():
            # Try to decode as UTF-8, but if that fails, use a special encoding
            # to preserve the exact byte sequence
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # For non-UTF-8 bytes, encode as escaped hex string
                # This ensures we can round-trip any byte sequence
                token_str = "\\x" + token_bytes.hex()
            vocab_data[token_str] = token_id
        
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # Save merges to txt
        with open(merges_path, "w", encoding="utf-8") as f:
            for first_bytes, second_bytes in self.merges:
                # Decode bytes to strings for text file, using hex encoding for non-UTF-8
                try:
                    first_str = first_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    first_str = "\\x" + first_bytes.hex()
                
                try:
                    second_str = second_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    second_str = "\\x" + second_bytes.hex()
                
                f.write(f"{first_str} {second_str}\n")
        
        self.logger.info(
            "BPE model saved",
            vocab_path=vocab_path,
            merges_path=merges_path,
            vocab_size=len(self.id_to_word),
            num_merges=len(self.merges)
        )


    def encode(self, text: str) -> list[int]:
        """Encodes the input text into a sequence of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        # 1. Split the input text into chunks based on special tokens.
        # Chunks are either normal text or special tokens.
        chunks: list[Chunk] = self._split_text_into_chunks(text)

        # 2. For each text chunk, pre-tokenize it into a list of words.
        # Special tokens are treated as single words.
        words_list = self._pre_tokenize_chunks(text, chunks)

        # 3. For each word, apply BPE merges and get the final token IDs.
        merged_result = self._bpe_and_encode_words(words_list)
        return merged_result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encodes an iterable of strings into a sequence of token IDs.

        Args:
            iterable (Iterable[str]): An iterable of strings to encode.

        Yields:
            Iterator[int]: A sequence of token IDs.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        all_bytes = b"".join(self.id_to_word.get(id_, b"") for id_ in ids)
        return all_bytes.decode(encoding="utf-8", errors="replace")

    def _split_text_into_chunks(self, text: str) -> list[Chunk]:
        """
        Split text on special tokens.
        Returns list of Chunk objects with start, end positions and associated special token.
        """
        if not self.special_tokens or len(self.special_tokens) == 0:
            return [Chunk(0, len(text), None)] if text else []

        # To handle overlapping special tokens correctly (e.g., "<|endoftext|>" and "<|endoftext|><|endoftext|>"),
        # we sort them by length in descending order. This ensures the longest match is found first.
        escaped_tokens = sorted(
            [re.escape(token) for token in self.special_tokens], key=len, reverse=True
        )
        pattern = "|".join(escaped_tokens)

        matches = re.finditer(pattern, text)

        token_positions = [
            (match.start(), match.end(), match.group()) for match in matches
        ]

        if not token_positions:
            return [Chunk(0, len(text), None)] if text else []

        chunks = []
        current_pos = 0

        for token_start, token_end, token_text in token_positions:
            # Add a text chunk for the content before the special token.
            if token_start > current_pos:
                chunks.append(Chunk(current_pos, token_start, None))

            # Add the special token as its own chunk.
            chunks.append(Chunk(token_start, token_end, token_text))
            current_pos = token_end

        # Add any remaining text after the last special token.
        if current_pos < len(text):
            chunks.append(Chunk(current_pos, len(text), None))

        return chunks

    def _pre_tokenize_chunks(self, text: str, chunks: list[Chunk]) -> list[list[str]]:
        words_list = []

        for chunk in chunks:
            words = []
            if chunk.special_token is not None:
                # If the chunk is a special token, it's treated as a single word.
                words.append(chunk.special_token)
            else:
                # For normal text chunks, pre-tokenize using the regex pattern.
                chunk_text = text[chunk.start : chunk.end]
                for match in re.finditer(self.pat, chunk_text):
                    words.append(match.group())

            if words:
                words_list.append(words)

        return words_list

    def _merge_word(self, word_str: str) -> Word:
        word_bytes = word_str.encode("utf-8")
        w = Word()
        for val_int in word_bytes:
            # Convert single byte to bytes object for lookup
            b = bytes([val_int])
            if b in self.word_to_id:
                byte_len = 1
                w.add(self.word_to_id[b], byte_len)
            else:
                # Log error before raising assertion
                self.logger.error(
                    "Unexpected byte in word",
                    byte_value=val_int,
                    byte_hex=f"0x{val_int:02x}",
                    byte_repr=repr(b),
                    word=word_str,
                )
                assert (
                    False
                ), f"Unexpected byte: {val_int} (0x{val_int:02x}) as {b!r} in decoding word: {word_str!r}"

        w.merge_all(self.merge_map)
        return w

    def _process_word(self, word: Word) -> list[int]:
        """Process a word and return its encoded representation.

        Args:
            word (Word): The word to process.

        Returns:
            list[int]: The encoded representation of the word.
        """
        encoded = []
        for symbol in word.symbols:
            if symbol.len == 0:
                continue
            if symbol.c not in self.id_to_word:
                self.logger.error(
                    "Symbol ID not found in vocabulary", symbol_id=symbol.c
                )
            assert (
                symbol.c in self.id_to_word
            ), f"Symbol ID {symbol.c} not found in vocabulary."
            encoded.append(symbol.c)
        return encoded

    def _bpe_and_encode_words(self, words_list: list[list[str]]) -> list[int]:
        encoded_word = []
        for words in words_list:
            for word in words:
                if self.special_tokens and word in self.special_tokens:
                    # Directly add special token ID
                    special_token_bytes = word.encode("utf-8")
                    if special_token_bytes in self.word_to_id:
                        encoded_word.append(self.word_to_id[special_token_bytes])
                    else:
                        self.logger.error(
                            "Special token not found in vocabulary", token=word
                        )
                        raise ValueError(
                            f"Special token {word} not found in vocabulary."
                        )
                else:
                    # Process regular word with BPE merges
                    encoded_word.extend(self._process_word(self._merge_word(word)))

        return encoded_word
