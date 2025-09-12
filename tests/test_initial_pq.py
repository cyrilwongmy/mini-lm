import pytest
import heapq
from cs336_basics.pretokenization import BPETokenizer, Merge


def test_empty_input():
    """Test initial_pq with empty input."""
    tokenizer = BPETokenizer()
    pair_counts = {}
    where_to_update = {}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    assert pq == []


def test_single_pair():
    """Test initial_pq with a single pair."""
    tokenizer = BPETokenizer()
    pair_counts = {(1, 2): 5}
    where_to_update = {(1, 2): {0, 1}}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    assert len(pq) == 1
    merge = pq[0]
    assert merge.pair == (1, 2)
    assert merge.count == 5
    assert merge.pos == {0, 1}


def test_multiple_pairs_different_counts():
    """Test initial_pq with multiple pairs having different counts."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (1, 2): 10,
        (3, 4): 5,
        (5, 6): 15
    }
    where_to_update = {
        (1, 2): {0},
        (3, 4): {1},
        (5, 6): {2}
    }
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    assert len(pq) == 3
    
    # Extract all merges and sort to check order
    merges = []
    while pq:
        merges.append(heapq.heappop(pq))
    
    # Should be ordered by count descending: (5,6):15, (1,2):10, (3,4):5
    assert merges[0].pair == (5, 6) and merges[0].count == 15
    assert merges[1].pair == (1, 2) and merges[1].count == 10
    assert merges[2].pair == (3, 4) and merges[2].count == 5


def test_tie_breaking_by_first_element():
    """Test initial_pq tie-breaking by first element of pair (descending)."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (1, 2): 10,
        (3, 4): 10,  # Same count as (1,2)
        (2, 1): 10   # Same count, intermediate first element
    }
    where_to_update = {
        (1, 2): {0},
        (3, 4): {1},
        (2, 1): {2}
    }
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Extract all merges
    merges = []
    while pq:
        merges.append(heapq.heappop(pq))
    
    # Should be ordered by first element descending: (3,4), (2,1), (1,2)
    assert merges[0].pair == (3, 4)
    assert merges[1].pair == (2, 1)
    assert merges[2].pair == (1, 2)
    # All should have same count
    assert all(merge.count == 10 for merge in merges)


def test_tie_breaking_by_second_element():
    """Test initial_pq tie-breaking by second element when first elements are equal."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (5, 1): 10,
        (5, 3): 10,  # Same count and first element as (5,1)
        (5, 2): 10   # Same count and first element, intermediate second element
    }
    where_to_update = {
        (5, 1): {0},
        (5, 3): {1},
        (5, 2): {2}
    }
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Extract all merges
    merges = []
    while pq:
        merges.append(heapq.heappop(pq))
    
    # Should be ordered by second element descending: (5,3), (5,2), (5,1)
    assert merges[0].pair == (5, 3)
    assert merges[1].pair == (5, 2)
    assert merges[2].pair == (5, 1)
    # All should have same count and first element
    assert all(merge.count == 10 and merge.pair[0] == 5 for merge in merges)


def test_complex_sorting():
    """Test initial_pq with complex sorting involving multiple tie-breaking levels."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (1, 1): 5,   # Lowest count
        (2, 3): 10,  # Medium count
        (2, 1): 10,  # Same count as (2,3), same first element, lower second element
        (3, 1): 10,  # Same count, higher first element
        (1, 2): 15   # Highest count
    }
    where_to_update = {pair: {i} for i, pair in enumerate(pair_counts.keys())}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Extract all merges
    merges = []
    while pq:
        merges.append(heapq.heappop(pq))
    
    # Expected order:
    # 1. (1,2): count=15 (highest count)
    # 2. (3,1): count=10 (highest first element among count=10)
    # 3. (2,3): count=10 (same first element as (2,1), higher second element)
    # 4. (2,1): count=10 (same first element as (2,3), lower second element)
    # 5. (1,1): count=5 (lowest count)
    
    expected_order = [(1, 2), (3, 1), (2, 3), (2, 1), (1, 1)]
    actual_order = [merge.pair for merge in merges]
    
    assert actual_order == expected_order


def test_heap_property_maintained():
    """Test that the returned structure maintains heap property."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (1, 1): 5,
        (2, 2): 10,
        (3, 3): 15,
        (4, 4): 8,
        (5, 5): 12
    }
    where_to_update = {pair: {i} for i, pair in enumerate(pair_counts.keys())}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Verify it's a proper heap by checking that popping elements gives sorted order
    sorted_merges = []
    pq_copy = pq.copy()
    while pq_copy:
        sorted_merges.append(heapq.heappop(pq_copy))
    
    # Verify descending order by count (primary sort key)
    for i in range(len(sorted_merges) - 1):
        current = sorted_merges[i]
        next_merge = sorted_merges[i + 1]
        # Current should have >= count than next (or be lexicographically greater if tied)
        assert (current.count, current.pair[0], current.pair[1]) >= \
               (next_merge.count, next_merge.pair[0], next_merge.pair[1])


def test_return_type_and_structure():
    """Test that initial_pq returns correct types and structure."""
    tokenizer = BPETokenizer()
    pair_counts = {(1, 2): 5, (3, 4): 10}
    where_to_update = {(1, 2): {0}, (3, 4): {1, 2}}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Check return type
    assert isinstance(pq, list)
    
    # Check element types
    assert all(isinstance(merge, Merge) for merge in pq)
    
    # Check that all original data is preserved
    pairs_in_pq = {merge.pair for merge in pq}
    assert pairs_in_pq == set(pair_counts.keys())
    
    for merge in pq:
        assert merge.count == pair_counts[merge.pair]
        assert merge.pos == where_to_update[merge.pair]


def test_large_numbers():
    """Test initial_pq with large numbers."""
    tokenizer = BPETokenizer()
    large_num = 1000000
    pair_counts = {
        (large_num, large_num + 1): 999999,
        (large_num + 2, large_num + 3): 1000000
    }
    where_to_update = {
        (large_num, large_num + 1): {0},
        (large_num + 2, large_num + 3): {1}
    }
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Higher count should come first
    first = heapq.heappop(pq)
    second = heapq.heappop(pq)
    
    assert first.pair == (large_num + 2, large_num + 3)
    assert first.count == 1000000
    assert second.pair == (large_num, large_num + 1)
    assert second.count == 999999


def test_zero_counts():
    """Test initial_pq with zero counts."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (1, 2): 0,
        (3, 4): 5,
        (5, 6): 0
    }
    where_to_update = {
        (1, 2): {0},
        (3, 4): {1},
        (5, 6): {2}
    }
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Extract all merges
    merges = []
    while pq:
        merges.append(heapq.heappop(pq))
    
    # Non-zero count should come first
    assert merges[0].count == 5
    # Zero counts should be ordered lexicographically (descending)
    zero_count_pairs = [merge.pair for merge in merges[1:]]
    assert (5, 6) in zero_count_pairs
    assert (1, 2) in zero_count_pairs
    # (5,6) should come before (1,2) due to lexicographic ordering
    assert zero_count_pairs.index((5, 6)) < zero_count_pairs.index((1, 2))


def test_identical_pairs_different_positions():
    """Test initial_pq with same pair appearing with different position sets."""
    tokenizer = BPETokenizer()
    # This shouldn't happen in normal usage, but test the function's behavior
    pair_counts = {(1, 2): 10}
    where_to_update = {(1, 2): {0, 1, 2}}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    assert len(pq) == 1
    merge = pq[0]
    assert merge.pair == (1, 2)
    assert merge.count == 10
    assert merge.pos == {0, 1, 2}


def test_heapify_correctness():
    """Test that heapify is working correctly by comparing with manual sorting."""
    tokenizer = BPETokenizer()
    pair_counts = {
        (i, j): (i * j) % 17  # Some pseudo-random counts
        for i in range(1, 6) for j in range(1, 4)
    }
    where_to_update = {pair: {hash(pair) % 10} for pair in pair_counts}
    
    pq = tokenizer.initial_pq(pair_counts, where_to_update)
    
    # Extract all elements from heap
    heap_order = []
    pq_copy = pq.copy()
    while pq_copy:
        heap_order.append(heapq.heappop(pq_copy))
    
    # Create manual sorted list
    manual_merges = [Merge(pair, count, where_to_update[pair]) 
                    for pair, count in pair_counts.items()]
    manual_merges.sort()  # Uses the __lt__ method
    
    # Both should give same order
    assert len(heap_order) == len(manual_merges)
    for heap_merge, manual_merge in zip(heap_order, manual_merges):
        assert heap_merge.pair == manual_merge.pair
        assert heap_merge.count == manual_merge.count
        assert heap_merge.pos == manual_merge.pos
