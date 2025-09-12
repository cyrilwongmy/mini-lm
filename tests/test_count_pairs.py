import pytest
from cs336_basics.pretokenization import BPETokenizer
from cs336_basics.word import Word, Symbol


def create_word_from_symbols(symbol_ids: list[int]) -> Word:
    """Helper function to create a Word with given symbol IDs."""
    word = Word()
    for symbol_id in symbol_ids:
        word.add(symbol_id, 1)
    return word


def test_empty_words():
    """Test count_pairs with empty word list."""
    tokenizer = BPETokenizer()
    words = []
    counts = []
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    assert pair_counts == {}
    assert where_to_update == {}


def test_single_word_single_symbol():
    """Test count_pairs with a single word containing one symbol."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65])]  # Single symbol 'A'
    counts = [1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # No pairs possible with single symbol
    assert pair_counts == {}
    assert where_to_update == {}


def test_single_word_two_symbols():
    """Test count_pairs with a single word containing two symbols."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65, 66])]  # 'AB'
    counts = [1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    expected_pair = (65, 66)
    assert pair_counts[expected_pair] == 1
    assert where_to_update[expected_pair] == {0}


def test_single_word_multiple_symbols():
    """Test count_pairs with a single word containing multiple symbols."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65, 66, 67, 68])]  # 'ABCD'
    counts = [1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    expected_pairs = [(65, 66), (66, 67), (67, 68)]  # AB, BC, CD
    
    assert len(pair_counts) == 3
    for pair in expected_pairs:
        assert pair_counts[pair] == 1
        assert where_to_update[pair] == {0}


def test_multiple_words_no_shared_pairs():
    """Test count_pairs with multiple words that don't share pairs."""
    tokenizer = BPETokenizer()
    words = [
        create_word_from_symbols([65, 66]),  # 'AB'
        create_word_from_symbols([67, 68])   # 'CD'
    ]
    counts = [1, 1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    assert pair_counts[(65, 66)] == 1
    assert pair_counts[(67, 68)] == 1
    assert where_to_update[(65, 66)] == {0}
    assert where_to_update[(67, 68)] == {1}


def test_multiple_words_shared_pairs():
    """Test count_pairs with multiple words that share pairs."""
    tokenizer = BPETokenizer()
    words = [
        create_word_from_symbols([65, 66]),  # 'AB'
        create_word_from_symbols([65, 66])   # 'AB' again
    ]
    counts = [1, 1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    assert pair_counts[(65, 66)] == 2  # Sum of counts
    assert where_to_update[(65, 66)] == {0, 1}  # Both word indices


def test_word_frequencies():
    """Test count_pairs with different word frequencies."""
    tokenizer = BPETokenizer()
    words = [
        create_word_from_symbols([65, 66]),  # 'AB'
        create_word_from_symbols([65, 66])   # 'AB' again
    ]
    counts = [3, 5]  # Different frequencies
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    assert pair_counts[(65, 66)] == 8  # 3 + 5
    assert where_to_update[(65, 66)] == {0, 1}


def test_overlapping_pairs_in_word():
    """Test count_pairs with overlapping pairs within a single word."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65, 65, 65])]  # 'AAA'
    counts = [2]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Should have pairs (A,A) appearing twice in the word, weighted by count
    assert pair_counts[(65, 65)] == 4  # 2 pairs * count of 2
    assert where_to_update[(65, 65)] == {0}


def test_complex_word_structure():
    """Test count_pairs with complex word containing repeated patterns."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65, 66, 65, 66, 67])]  # 'ABABC'
    counts = [1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    expected_pairs = {
        (65, 66): 2,  # AB appears twice
        (66, 65): 1,  # BA appears once
        (66, 67): 1   # BC appears once
    }
    
    assert len(pair_counts) == 3
    for pair, expected_count in expected_pairs.items():
        assert pair_counts[pair] == expected_count
        assert where_to_update[pair] == {0}


def test_multiple_words_complex_sharing():
    """Test count_pairs with multiple words having complex pair sharing."""
    tokenizer = BPETokenizer()
    words = [
        create_word_from_symbols([65, 66, 67]),  # 'ABC'
        create_word_from_symbols([66, 67, 68]),  # 'BCD'
        create_word_from_symbols([65, 66, 68])   # 'ABD'
    ]
    counts = [2, 3, 1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    expected_results = {
        (65, 66): (3, {0, 2}),  # AB: 2 from word 0, 1 from word 2
        (66, 67): (5, {0, 1}),  # BC: 2 from word 0, 3 from word 1
        (67, 68): (3, {1}),     # CD: 3 from word 1
        (66, 68): (1, {2})      # BD: 1 from word 2
    }
    
    assert len(pair_counts) == 4
    for pair, (expected_count, expected_indices) in expected_results.items():
        assert pair_counts[pair] == expected_count
        assert where_to_update[pair] == expected_indices


def test_large_symbol_ids():
    """Test count_pairs with large symbol IDs."""
    tokenizer = BPETokenizer()
    large_id1 = 1000000
    large_id2 = 2000000
    
    words = [create_word_from_symbols([large_id1, large_id2])]
    counts = [1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    assert pair_counts[(large_id1, large_id2)] == 1
    assert where_to_update[(large_id1, large_id2)] == {0}


def test_zero_counts():
    """Test count_pairs with zero counts."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65, 66])]
    counts = [0]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Should still track the pair but with zero count
    assert pair_counts[(65, 66)] == 0
    assert where_to_update[(65, 66)] == {0}


def test_return_types():
    """Test that count_pairs returns the correct types."""
    tokenizer = BPETokenizer()
    words = [create_word_from_symbols([65, 66])]
    counts = [1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Check return types
    assert isinstance(pair_counts, dict)
    assert isinstance(where_to_update, dict)
    
    # Check key and value types
    for pair, count in pair_counts.items():
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(pair[0], int)
        assert isinstance(pair[1], int)
        assert isinstance(count, int)
    
    for pair, indices in where_to_update.items():
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(indices, set)
        assert all(isinstance(idx, int) for idx in indices)


def test_consistent_pair_tracking():
    """Test that pair_counts and where_to_update are consistent."""
    tokenizer = BPETokenizer()
    words = [
        create_word_from_symbols([65, 66, 67]),
        create_word_from_symbols([66, 67, 68]),
        create_word_from_symbols([65, 66])
    ]
    counts = [2, 3, 1]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Every pair in pair_counts should be in where_to_update and vice versa
    assert set(pair_counts.keys()) == set(where_to_update.keys())
    
    # For each pair, verify the count matches the sum of counts for tracked words
    for pair in pair_counts:
        expected_count = sum(counts[i] for i in where_to_update[pair])
        assert pair_counts[pair] == expected_count


def test_integration_with_tokenize_words():
    """Test count_pairs with output from tokenize_words."""
    tokenizer = BPETokenizer()
    
    # Create word counts and mappings
    wc = {b'hello': 2, b'world': 1}
    w2id = {}
    id2w = {}
    tokenizer.compute_alphabet(w2id, id2w)
    
    # Get words from tokenize_words
    words, counts = tokenizer.tokenize_words(wc, w2id)
    
    # Count pairs
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Verify basic properties
    assert isinstance(pair_counts, dict)
    assert isinstance(where_to_update, dict)
    assert len(pair_counts) > 0  # Should have some pairs
    assert set(pair_counts.keys()) == set(where_to_update.keys())
    
    # Verify all pairs use valid symbol IDs
    for pair in pair_counts:
        assert pair[0] in id2w
        assert pair[1] in id2w


def test_empty_words_in_list():
    """Test count_pairs when some words have no symbols."""
    tokenizer = BPETokenizer()
    words = [
        Word(),  # Empty word
        create_word_from_symbols([65, 66]),  # 'AB'
        Word()   # Another empty word
    ]
    counts = [1, 2, 3]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Only the non-empty word should contribute pairs
    assert pair_counts[(65, 66)] == 2
    assert where_to_update[(65, 66)] == {1}


def test_single_symbol_words_mixed():
    """Test count_pairs with mix of single-symbol and multi-symbol words."""
    tokenizer = BPETokenizer()
    words = [
        create_word_from_symbols([65]),        # Single symbol 'A'
        create_word_from_symbols([66, 67]),    # 'BC'
        create_word_from_symbols([68]),        # Single symbol 'D'
        create_word_from_symbols([66, 67])     # 'BC' again
    ]
    counts = [1, 2, 1, 3]
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    # Only multi-symbol words contribute pairs
    assert pair_counts[(66, 67)] == 5  # 2 + 3
    assert where_to_update[(66, 67)] == {1, 3}
    assert len(pair_counts) == 1


def test_performance_with_many_words():
    """Test count_pairs performance with many words."""
    tokenizer = BPETokenizer()
    
    # Create many words with same pattern
    num_words = 1000
    words = [create_word_from_symbols([65, 66]) for _ in range(num_words)]
    counts = [1] * num_words
    
    pair_counts, where_to_update = tokenizer.count_pairs(words, counts)
    
    assert pair_counts[(65, 66)] == num_words
    assert len(where_to_update[(65, 66)]) == num_words
    assert where_to_update[(65, 66)] == set(range(num_words))
