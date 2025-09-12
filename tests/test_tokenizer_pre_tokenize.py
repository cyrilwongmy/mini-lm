import pytest
from cs336_basics.tokenizer import Tokenizer


class TestPreTokenize:
    """Test suite for the pre_tokenize method of Tokenizer class."""

    def test_no_special_tokens(self):
        """Test pre_tokenize with no special tokens in text."""
        vocab = {0: b"hello", 1: b"world"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello world"
        result = tokenizer.find_chunk(text)

        # Should return single chunk covering entire text
        assert result == [(0, 11)]

    def test_single_special_token(self):
        """Test pre_tokenize with one special token occurrence."""
        vocab = {0: b"hello", 1: b"world", 2: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello<|endoftext|>world"
        result = tokenizer.find_chunk(text)

        # Should split into two chunks: "hello" and "world"
        assert result == [(0, 5), (18, 23)]

    def test_multiple_special_tokens(self):
        """Test pre_tokenize with multiple special token occurrences."""
        vocab = {0: b"hello", 1: b"world", 2: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello<|endoftext|>world<|endoftext|>foo"
        result = tokenizer.find_chunk(text)

        # Should split into three chunks
        assert result == [(0, 5), (18, 23), (36, 39)]

    def test_consecutive_special_tokens(self):
        """Test pre_tokenize with consecutive special tokens."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello<|endoftext|><|endoftext|>world"
        result = tokenizer.find_chunk(text)

        # Should handle consecutive tokens correctly
        assert result == [(0, 5), (31, 36)]

    def test_special_token_at_start(self):
        """Test pre_tokenize with special token at the beginning."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "<|endoftext|>hello world"
        result = tokenizer.find_chunk(text)

        # Should return chunk after the special token
        assert result == [(13, 24)]

    def test_special_token_at_end(self):
        """Test pre_tokenize with special token at the end."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello world<|endoftext|>"
        result = tokenizer.find_chunk(text)

        # Should return chunk before the special token
        assert result == [(0, 11)]

    def test_empty_string(self):
        """Test pre_tokenize with empty string."""
        vocab = {0: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = ""
        result = tokenizer.find_chunk(text)

        # Should return empty list for empty string
        assert result == []

    def test_only_special_token(self):
        """Test pre_tokenize with only special token."""
        vocab = {0: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "<|endoftext|>"
        result = tokenizer.find_chunk(text)

        # Should return empty list when text is only special token
        assert result == []

    def test_overlapping_special_tokens(self):
        """Test pre_tokenize with overlapping occurrences of special token."""
        vocab = {0: b"aaa"}
        merges = []
        special_tokens = ["aa"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "aaa"
        result = tokenizer.find_chunk(text)

        # The implementation finds "aa" at position 0, then continues from position 2
        # So it only finds one occurrence, not overlapping ones
        assert (
            result == []
        )  # No chunks because "aa" at 0 leaves only "a" which starts at 2

    def test_special_token_with_spaces(self):
        """Test pre_tokenize with special token containing spaces."""
        vocab = {0: b"hello", 1: b"< | >"}
        merges = []
        special_tokens = ["< | >"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello< | >world< | >end"
        result = tokenizer.find_chunk(text)

        # Should split correctly with space-containing special token
        assert result == [(0, 5), (10, 15), (20, 23)]

    def test_unicode_text_with_special_token(self):
        """Test pre_tokenize with Unicode text."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "café<|endoftext|>naïve"
        result = tokenizer.find_chunk(text)

        # Should handle Unicode correctly
        # "café" has 4 characters, "naïve" has 5 characters
        assert result == [(0, 4), (17, 22)]

    def test_long_text_with_many_tokens(self):
        """Test pre_tokenize with long text containing many special tokens."""
        vocab = {0: b"text", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        # Create text with 100 segments
        segments = ["segment" + str(i) for i in range(100)]
        text = "<|endoftext|>".join(segments)
        result = tokenizer.find_chunk(text)

        # Should have 100 chunks
        assert len(result) == 100

        # Verify first and last chunks
        assert text[result[0][0] : result[0][1]] == "segment0"
        assert text[result[-1][0] : result[-1][1]] == "segment99"

    def test_special_token_not_in_vocab(self):
        """Test pre_tokenize when special token is not in vocab."""
        vocab = {0: b"hello", 1: b"world"}
        merges = []
        special_tokens = ["<|special|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello<|special|>world"
        result = tokenizer.find_chunk(text)

        # Should still split correctly even if special token not in vocab
        assert result == [(0, 5), (16, 21)]

    def test_assertion_multiple_special_tokens(self):
        """Test that assertion is raised when multiple special tokens are provided."""
        vocab = {0: b"hello"}
        merges = []
        special_tokens = ["<|endoftext|>", "<|startoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        with pytest.raises(AssertionError, match="Only one special token is supported"):
            tokenizer.find_chunk("hello world")

    def test_no_special_tokens_list(self):
        """Test pre_tokenize when no special tokens are provided."""
        vocab = {0: b"hello"}
        merges = []
        special_tokens = []
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        # Should raise assertion error for empty special tokens list
        with pytest.raises(AssertionError, match="Only one special token is supported"):
            tokenizer.find_chunk("hello world")

    def test_chunk_boundaries_correctness(self):
        """Test that chunk boundaries correctly reconstruct the original text."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello<|endoftext|>world<|endoftext|>foo bar<|endoftext|>"
        result = tokenizer.find_chunk(text)

        # Extract chunks using boundaries
        chunks = [text[start:end] for start, end in result]

        # Verify chunks are correct
        assert chunks == ["hello", "world", "foo bar"]

        # Verify no overlap between chunks
        for i in range(len(result) - 1):
            assert result[i][1] <= result[i + 1][0]

    def test_special_token_partial_match(self):
        """Test that partial matches of special token are not treated as special tokens."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        text = "hello <|endof text|> world"
        result = tokenizer.find_chunk(text)

        # Should return single chunk as no exact match found
        # Text length is 26 characters
        assert result == [(0, 26)]

    def test_performance_with_no_special_tokens(self):
        """Test performance when text contains no special tokens."""
        vocab = {0: b"hello", 1: b"<|endoftext|>"}
        merges = []
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer(vocab, merges, special_tokens)

        # Large text with no special tokens
        text = "a" * 10000
        result = tokenizer.find_chunk(text)

        # Should return single chunk efficiently
        assert result == [(0, 10000)]
        assert len(result) == 1
