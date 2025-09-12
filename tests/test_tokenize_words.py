import pytest
from cs336_basics.pretokenization import BPETokenizer
from cs336_basics.word import Word, Symbol


def test_empty_word_counts():
    """Test tokenize_words with empty word counts."""
    tokenizer = BPETokenizer()
    wc = {}
    w2id = {}
    id2w = {}
    
    # First compute the alphabet to populate w2id and id2w
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert words == []
    assert counts == []


def test_single_word_single_byte():
    """Test tokenize_words with a single word containing one byte."""
    tokenizer = BPETokenizer()
    wc = {b'A': 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == 1
    
    # Check the word structure
    word = words[0]
    assert len(word.symbols) == 1
    assert word.symbols[0].c == w2id[b'A']
    assert word.symbols[0].len == 1
    assert word.symbols[0].prev == -1
    assert word.symbols[0].next == -1


def test_single_word_multiple_bytes():
    """Test tokenize_words with a single word containing multiple bytes."""
    tokenizer = BPETokenizer()
    wc = {b'ABC': 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == 1
    
    # Check the word structure
    word = words[0]
    assert len(word.symbols) == 3
    
    # Check first symbol (A)
    assert word.symbols[0].c == w2id[bytes([ord('A')])]
    assert word.symbols[0].len == 1
    assert word.symbols[0].prev == -1
    assert word.symbols[0].next == -1
    
    # Check second symbol (B)
    assert word.symbols[1].c == w2id[bytes([ord('B')])]
    assert word.symbols[1].len == 1
    assert word.symbols[1].prev == 0
    assert word.symbols[1].next == -1
    
    # Check third symbol (C)
    assert word.symbols[2].c == w2id[bytes([ord('C')])]
    assert word.symbols[2].len == 1
    assert word.symbols[2].prev == 1
    assert word.symbols[2].next == -1


def test_multiple_words_different_counts():
    """Test tokenize_words with multiple words having different counts."""
    tokenizer = BPETokenizer()
    wc = {b'A': 3, b'B': 1, b'AB': 2}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 3
    assert len(counts) == 3
    
    # The order depends on dictionary iteration, so we need to check by matching
    word_count_pairs = list(zip(words, counts))
    
    # Find the single-byte words
    single_byte_words = [(w, c) for w, c in word_count_pairs if len(w.symbols) == 1]
    double_byte_words = [(w, c) for w, c in word_count_pairs if len(w.symbols) == 2]
    
    assert len(single_byte_words) == 2  # 'A' and 'B'
    assert len(double_byte_words) == 1  # 'AB'
    
    # Check counts
    single_byte_counts = [c for w, c in single_byte_words]
    assert 3 in single_byte_counts  # count for 'A'
    assert 1 in single_byte_counts  # count for 'B'
    
    double_byte_count = double_byte_words[0][1]
    assert double_byte_count == 2  # count for 'AB'


def test_utf8_encoded_words():
    """Test tokenize_words with UTF-8 encoded words."""
    tokenizer = BPETokenizer()
    # UTF-8 encoding of 'café' is b'caf\xc3\xa9'
    cafe_bytes = 'café'.encode('utf-8')
    wc = {cafe_bytes: 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == 1
    
    # Check the word structure - should have 5 symbols (c, a, f, é as two bytes)
    word = words[0]
    assert len(word.symbols) == 5
    
    # Verify each byte is correctly mapped
    expected_bytes = [b'c', b'a', b'f', b'\xc3', b'\xa9']
    for i, expected_byte in enumerate(expected_bytes):
        assert word.symbols[i].c == w2id[expected_byte]
        assert word.symbols[i].len == 1


def test_word_with_special_tokens():
    """Test tokenize_words with special tokens added to the vocabulary."""
    tokenizer = BPETokenizer()
    wc = {b'hello': 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet and add special tokens
    tokenizer.compute_alphabet(w2id, id2w)
    tokenizer.add_special_tokens(w2id, id2w, ['<|endoftext|>'])
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == 1
    
    # Check the word structure
    word = words[0]
    assert len(word.symbols) == 5  # h, e, l, l, o
    
    # Verify each byte is correctly mapped
    expected_chars = [b'h', b'e', b'l', b'l', b'o']
    for i, expected_char in enumerate(expected_chars):
        assert word.symbols[i].c == w2id[expected_char]
        assert word.symbols[i].len == 1


def test_words_with_zero_bytes():
    """Test tokenize_words with words containing zero bytes."""
    tokenizer = BPETokenizer()
    wc = {b'\x00\x01\x02': 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == 1
    
    # Check the word structure
    word = words[0]
    assert len(word.symbols) == 3
    
    # Verify each byte is correctly mapped
    expected_bytes = [bytes([0]), bytes([1]), bytes([2])]
    for i, expected_byte in enumerate(expected_bytes):
        assert word.symbols[i].c == w2id[expected_byte]
        assert word.symbols[i].len == 1


def test_large_word_counts():
    """Test tokenize_words with large count values."""
    tokenizer = BPETokenizer()
    large_count = 1000000
    wc = {b'test': large_count}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == large_count


def test_return_types():
    """Test that tokenize_words returns the correct types."""
    tokenizer = BPETokenizer()
    wc = {b'test': 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    # Check return types
    assert isinstance(words, list)
    assert isinstance(counts, list)
    assert len(words) == len(counts)
    
    # Check element types
    assert all(isinstance(word, Word) for word in words)
    assert all(isinstance(count, int) for count in counts)
    assert all(count > 0 for count in counts)


def test_word_symbols_linking():
    """Test that word symbols are properly linked with prev/next pointers."""
    tokenizer = BPETokenizer()
    wc = {b'ABCD': 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    word = words[0]
    assert len(word.symbols) == 4
    
    # Check linking structure
    # First symbol: prev=-1, next=-1 (as per Word.add implementation)
    assert word.symbols[0].prev == -1
    assert word.symbols[0].next == -1
    
    # Second symbol: prev=0, next=-1
    assert word.symbols[1].prev == 0
    assert word.symbols[1].next == -1
    
    # Third symbol: prev=1, next=-1
    assert word.symbols[2].prev == 1
    assert word.symbols[2].next == -1
    
    # Fourth symbol: prev=2, next=-1
    assert word.symbols[3].prev == 2
    assert word.symbols[3].next == -1


def test_consistency_with_pretokenization():
    """Test tokenize_words with output from pre_tokenization to ensure consistency."""
    tokenizer = BPETokenizer()
    
    # Use pre_tokenization to get word counts
    text = "hello world test"
    wc = tokenizer.pre_tokenization(text)
    
    w2id = {}
    id2w = {}
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    # Should have same number of unique words
    assert len(words) == len(wc)
    assert len(counts) == len(wc)
    
    # Total count should match
    assert sum(counts) == sum(wc.values())
    
    # Each word should be properly tokenized
    for word in words:
        assert len(word.symbols) > 0
        assert all(isinstance(symbol, Symbol) for symbol in word.symbols)


def test_all_byte_values():
    """Test tokenize_words with all possible byte values."""
    tokenizer = BPETokenizer()
    
    # Create a word with all 256 possible byte values
    all_bytes = bytes(range(256))
    wc = {all_bytes: 1}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 1
    assert len(counts) == 1
    assert counts[0] == 1
    
    # Check the word structure
    word = words[0]
    assert len(word.symbols) == 256
    
    # Verify each byte is correctly mapped
    for i in range(256):
        expected_byte = bytes([i])
        assert word.symbols[i].c == w2id[expected_byte]
        assert word.symbols[i].len == 1


def test_order_preservation():
    """Test that the order of words and counts is preserved."""
    tokenizer = BPETokenizer()
    
    # Use an ordered input to test preservation
    wc = {b'first': 1, b'second': 2, b'third': 3}
    w2id = {}
    id2w = {}
    
    # Compute alphabet to populate mappings
    tokenizer.compute_alphabet(w2id, id2w)
    
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    assert len(words) == 3
    assert len(counts) == 3
    
    # Create a mapping from reconstructed words to their counts
    word_to_count = {}
    for word, count in zip(words, counts):
        # Reconstruct the original bytes from the word symbols
        reconstructed = b''.join(id2w[symbol.c] for symbol in word.symbols)
        word_to_count[reconstructed] = count
    
    # Verify all original words are present with correct counts
    for orig_word, orig_count in wc.items():
        assert orig_word in word_to_count
        assert word_to_count[orig_word] == orig_count
