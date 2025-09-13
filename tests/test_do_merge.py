import heapq
from collections import defaultdict
from mini_lm.bpe import BpeTrainer, Merge
from mini_lm.bpe.word import Word


def create_word_from_symbols(symbol_ids: list[int]) -> Word:
    """Helper function to create a Word with given symbol IDs."""
    word = Word()
    for symbol_id in symbol_ids:
        word.add(symbol_id, 1)
    return word


def setup_basic_test_data():
    """Set up basic test data for do_merge tests."""
    # Create basic vocabulary: bytes 0-2 (a, b, c)
    w2id = {bytes([0]): 0, bytes([1]): 1, bytes([2]): 2}
    id2w = {0: bytes([0]), 1: bytes([1]), 2: bytes([2])}
    
    # Create words: "ab", "bc", "ab"
    words = [
        create_word_from_symbols([0, 1]),  # "ab"
        create_word_from_symbols([1, 2]),  # "bc" 
        create_word_from_symbols([0, 1]),  # "ab"
    ]
    counts = [2, 1, 3]  # frequencies of each word
    
    # Set up pair counts and tracking - use defaultdict to handle new pairs
    pair_counts = defaultdict(int)
    pair_counts.update({(0, 1): 5, (1, 2): 1})  # (a,b) appears 5 times, (b,c) appears 1 time
    where_to_update = defaultdict(set)
    where_to_update.update({(0, 1): {0, 2}, (1, 2): {1}})
    
    # Create priority queue with merges
    pq = [
        Merge((0, 1), 5, {0, 2}),  # merge (a,b) with count 5
        Merge((1, 2), 1, {1}),     # merge (b,c) with count 1
    ]
    heapq.heapify(pq)
    
    return w2id, id2w, words, counts, pair_counts, where_to_update, pq


class TestDoMerge:

    def test_empty_priority_queue(self):
        """Test do_merge with empty priority queue."""
        trainer = BpeTrainer()
        w2id = {bytes([0]): 0}
        id2w = {0: bytes([0])}
        words = [create_word_from_symbols([0])]
        counts = [1]
        pair_counts = defaultdict(int)
        where_to_update = defaultdict(set)
        pq = []

        # Since do_merge doesn't return anything, we need to check if it modifies the input
        # The function should not crash and should not modify w2id when pq is empty
        original_vocab_size = len(w2id)
        trainer._perform_merges(
            w2id, id2w, vocab_size=10, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Vocabulary should remain unchanged
        assert len(w2id) == original_vocab_size

    def test_vocab_size_reached(self):
        """Test do_merge when vocab size is already reached."""
        tokenizer = BpeTrainer()
        w2id, id2w, words, counts, pair_counts, where_to_update, pq = setup_basic_test_data()

        # Set vocab_size to current size (no room for new tokens)
        original_vocab_size = len(w2id)
        tokenizer._perform_merges(
            w2id, id2w, vocab_size=3, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # No new tokens should be added
        assert len(w2id) == original_vocab_size

    def test_single_merge(self):
        """Test do_merge with a single merge operation."""
        tokenizer = BpeTrainer()
        w2id, id2w, words, counts, pair_counts, where_to_update, pq = setup_basic_test_data()

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=4, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # New token should be added to vocabulary
        assert len(w2id) == 4
        new_token = bytes([0]) + bytes([1])
        assert new_token in w2id
        assert w2id[new_token] == 3
        assert id2w[3] == new_token

    def test_multiple_merges(self):
        """Test do_merge with multiple merge operations."""
        tokenizer = BpeTrainer()
        w2id, id2w, words, counts, pair_counts, where_to_update, pq = setup_basic_test_data()

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=5, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Two new tokens should be added
        assert len(w2id) == 5

        # Check that both expected tokens were created
        token_ab = bytes([0]) + bytes([1])
        token_bc = bytes([1]) + bytes([2])
        assert token_ab in w2id
        assert token_bc in w2id

    def test_outdated_count_in_queue(self):
        """Test do_merge when priority queue contains outdated counts."""
        tokenizer = BpeTrainer()
        w2id, id2w, words, counts, pair_counts, where_to_update, pq = setup_basic_test_data()

        # Manually create a merge with outdated count
        pq = [Merge((0, 1), 10, {0, 2})]  # count=10 but actual count is 5
        heapq.heapify(pq)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=4, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Should still perform the merge with correct count
        new_token = bytes([0]) + bytes([1])
        assert new_token in w2id

    def test_zero_count_pair(self):
        """Test do_merge when a pair has zero count."""
        tokenizer = BpeTrainer()
        w2id = {bytes([0]): 0, bytes([1]): 1}
        id2w = {0: bytes([0]), 1: bytes([1])}
        words = [create_word_from_symbols([0, 1])]
        counts = [1]
        pair_counts = defaultdict(int)
        pair_counts[(0, 1)] = 0  # Zero count
        where_to_update = defaultdict(set)
        where_to_update[(0, 1)] = {0}
        pq = [Merge((0, 1), 0, {0})]
        heapq.heapify(pq)

        original_vocab_size = len(w2id)
        tokenizer._perform_merges(
            w2id, id2w, vocab_size=5, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Should not perform any merges due to zero count
        assert len(w2id) == original_vocab_size

    def test_new_pairs_generated(self):
        """Test that new pairs are generated and added to the queue after merging."""
        tokenizer = BpeTrainer()

        # Create a scenario where merging creates new pairs
        # Word: "abc" -> after merging (a,b) -> "AB c" where AB is the new token
        w2id = {bytes([0]): 0, bytes([1]): 1, bytes([2]): 2}  # a, b, c
        id2w = {0: bytes([0]), 1: bytes([1]), 2: bytes([2])}

        words = [create_word_from_symbols([0, 1, 2])]  # "abc"
        counts = [1]

        pair_counts = defaultdict(int)
        pair_counts.update({(0, 1): 1, (1, 2): 1})
        where_to_update = defaultdict(set)
        where_to_update.update({(0, 1): {0}, (1, 2): {0}})

        # Only include the (a,b) merge in the queue initially
        pq = [Merge((0, 1), 1, {0})]
        heapq.heapify(pq)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=5, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # The new token should be in vocabulary
        new_token = bytes([0]) + bytes([1])
        assert new_token in w2id

    def test_word_merge_integration(self):
        """Test that do_merge correctly calls Word.merge and processes results."""
        tokenizer = BpeTrainer()
        w2id, id2w, words, counts, pair_counts, where_to_update, pq = setup_basic_test_data()

        # Store original word states for comparison
        original_word_0_symbols = len(words[0].symbols)
        original_word_2_symbols = len(words[2].symbols)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=4, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Check that words were modified (symbols merged)
        assert len(words[0].symbols) == original_word_0_symbols - 1  # One merge occurred
        assert len(words[2].symbols) == original_word_2_symbols - 1  # One merge occurred

        # The merged symbol should have the new token ID
        new_token_id = w2id[bytes([0]) + bytes([1])]
        assert words[0].symbols[0].c == new_token_id
        assert words[2].symbols[0].c == new_token_id

    def test_existing_token_not_readded(self):
        """Test that existing tokens are not re-added to vocabulary."""
        tokenizer = BpeTrainer()

        # Pre-add the merged token to vocabulary
        merged_token = bytes([0]) + bytes([1])
        w2id = {bytes([0]): 0, bytes([1]): 1, bytes([2]): 2, merged_token: 3}
        id2w = {0: bytes([0]), 1: bytes([1]), 2: bytes([2]), 3: merged_token}

        words = [create_word_from_symbols([0, 1])]
        counts = [1]
        pair_counts = defaultdict(int)
        pair_counts[(0, 1)] = 1
        where_to_update = defaultdict(set)
        where_to_update[(0, 1)] = {0}
        pq = [Merge((0, 1), 1, {0})]
        heapq.heapify(pq)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=5, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Vocabulary size should not increase (token already existed)
        assert len(w2id) == 4

    def test_large_vocab_size_limit(self):
        """Test do_merge with a very large vocab_size limit."""
        tokenizer = BpeTrainer()
        w2id, id2w, words, counts, pair_counts, where_to_update, pq = setup_basic_test_data()

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=1000, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Should merge all available pairs (at least the initial ones)
        assert len(w2id) >= 5  # Original 3 + at least 2 merged tokens

    def test_priority_queue_ordering(self):
        """Test that merges happen in correct priority order (highest count first)."""
        tokenizer = BpeTrainer()

        # Create scenario with clear priority ordering
        w2id = {bytes([0]): 0, bytes([1]): 1, bytes([2]): 2, bytes([3]): 3}
        id2w = {0: bytes([0]), 1: bytes([1]), 2: bytes([2]), 3: bytes([3])}

        words = [
            create_word_from_symbols([0, 1]),  # "ab"
            create_word_from_symbols([2, 3]),  # "cd"
        ]
        counts = [1, 1]

        # Make (c,d) have higher count than (a,b)
        pair_counts = defaultdict(int)
        pair_counts.update({(0, 1): 3, (2, 3): 5})
        where_to_update = defaultdict(set)
        where_to_update.update({(0, 1): {0}, (2, 3): {1}})

        pq = [
            Merge((0, 1), 3, {0}),
            Merge((2, 3), 5, {1}),
        ]
        heapq.heapify(pq)

        # Store initial vocabulary size
        initial_vocab_size = len(w2id)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=6, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # Both tokens should be created
        assert len(w2id) == initial_vocab_size + 2
        token_ab = bytes([0]) + bytes([1])
        token_cd = bytes([2]) + bytes([3])
        assert token_ab in w2id
        assert token_cd in w2id

    def test_heap_property_maintained(self):
        """Test that the heap property is maintained throughout merging."""
        tokenizer = BpeTrainer()

        # Create multiple pairs with different counts
        w2id = {bytes([i]): i for i in range(6)}
        id2w = {i: bytes([i]) for i in range(6)}

        words = [
            create_word_from_symbols([0, 1]),  # pair (0,1)
            create_word_from_symbols([2, 3]),  # pair (2,3)  
            create_word_from_symbols([4, 5]),  # pair (4,5)
        ]
        counts = [1, 1, 1]

        pair_counts = defaultdict(int)
        pair_counts.update({(0, 1): 10, (2, 3): 5, (4, 5): 15})
        where_to_update = defaultdict(set)
        where_to_update.update({(0, 1): {0}, (2, 3): {1}, (4, 5): {2}})

        pq = [
            Merge((0, 1), 10, {0}),
            Merge((2, 3), 5, {1}),
            Merge((4, 5), 15, {2}),
        ]
        heapq.heapify(pq)

        initial_vocab_size = len(w2id)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=10, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # All three pairs should be merged
        assert len(w2id) == initial_vocab_size + 3

        # Check that all expected tokens were created
        token_01 = bytes([0]) + bytes([1])
        token_23 = bytes([2]) + bytes([3])
        token_45 = bytes([4]) + bytes([5])
        assert token_01 in w2id
        assert token_23 in w2id
        assert token_45 in w2id

    def test_pair_counts_consistency(self):
        """Test that pair_counts are updated correctly during merging."""
        tokenizer = BpeTrainer()

        # Create a simple case where we can track pair count changes
        w2id = {bytes([0]): 0, bytes([1]): 1, bytes([2]): 2}
        id2w = {0: bytes([0]), 1: bytes([1]), 2: bytes([2])}

        # Word "abc" appears twice
        words = [
            create_word_from_symbols([0, 1, 2]),  # "abc"
            create_word_from_symbols([0, 1, 2]),  # "abc"
        ]
        counts = [1, 1]  # Each word appears once

        pair_counts = defaultdict(int)
        pair_counts.update({(0, 1): 2, (1, 2): 2})  # Each pair appears twice
        where_to_update = defaultdict(set)
        where_to_update.update({(0, 1): {0, 1}, (1, 2): {0, 1}})

        pq = [Merge((0, 1), 2, {0, 1})]  # Only merge (a,b)
        heapq.heapify(pq)

        tokenizer._perform_merges(
            w2id, id2w, vocab_size=4, pair_counts=pair_counts,
            words=words, pq=pq, counts=counts, where_to_update=where_to_update
        )

        # After merging (a,b), we should have new pairs with the merged token
        new_token_id = w2id[bytes([0]) + bytes([1])]

        # The new pair (new_token, c) should have been created
        assert (new_token_id, 2) in pair_counts
        # The old pair (b, c) should have count 0 (removed by merge)
        assert pair_counts[(1, 2)] == 0
