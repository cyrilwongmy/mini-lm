from mini_lm.bpe.word import Word, Symbol


def test_symbol_initialization():
        """Test Symbol class initialization."""
        symbol = Symbol(c=65, prev=0, nxt=2, len=1)
        assert symbol.c == 65
        assert symbol.prev == 0
        assert symbol.next == 2
        assert symbol.len == 1


def test_word_initialization():
    """Test Word class initialization."""
    word = Word()
    assert word.symbols == []

def test_add_single_symbol():
    """Test adding a single symbol to an empty word."""
    word = Word()
    word.add(65, 1)  # 'A'
    
    assert len(word.symbols) == 1
    assert word.symbols[0].c == 65
    assert word.symbols[0].prev == -1
    assert word.symbols[0].next == -1
    assert word.symbols[0].len == 1

def test_add_multiple_symbols():
    """Test adding multiple symbols to a word."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    assert len(word.symbols) == 3
    
    # Check first symbol
    assert word.symbols[0].c == 65
    assert word.symbols[0].prev == -1
    assert word.symbols[0].next == 1
    
    # Check second symbol
    assert word.symbols[1].c == 66
    assert word.symbols[1].prev == 0
    assert word.symbols[1].next == 2
    
    # Check third symbol
    assert word.symbols[2].c == 67
    assert word.symbols[2].prev == 1
    assert word.symbols[2].next == -1

def test_merge_basic():
    """Test basic merge functionality."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    # Merge A and B into a new symbol 256
    changes = word.merge(65, 66, 256)
    
    # Should have 2 symbols after merge: [256, C]
    assert len(word.symbols) == 2
    assert word.symbols[0].c == 256
    assert word.symbols[0].len == 2  # Combined length
    assert word.symbols[1].c == 67

    # Check changes
    expected_changes = [
        ((65, 66), -1),  # Remove A-B pair
        ((66, 67), -1),  # Remove B-C pair
        ((256, 67), 1)   # Add new pair (AB, C) with next symbol
    ]
    assert changes == expected_changes

def test_merge_no_match():
    """Test merge when no matching pair exists."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    # Try to merge X and Y (not present)
    changes = word.merge(88, 89, 256)
    
    # Should have no changes
    assert len(word.symbols) == 3
    assert changes == []

def test_merge_triple_newlines_special_case():
    """Test merge function with the special case where original string is '\n\n\n'.
    It should only merge the first two '\n' and leave the last one alone.
    """
    word = Word()
    # UTF-8 newline token
    newline_token = "\n".encode("utf-8")
    # convert the bytes to int
    newline_token = int.from_bytes(newline_token, "big")
    
    # Add three newline symbols
    word.add(newline_token, 1)  # First '\n'
    word.add(newline_token, 1)  # Second '\n'
    word.add(newline_token, 1)  # Third '\n'
    
    # Merge the first two newlines into a double newline token
    double_newline_token = "\n\n".encode("utf-8")
    double_newline_token = int.from_bytes(double_newline_token, byteorder="big")
    changes = word.merge(newline_token, newline_token, double_newline_token)
    
    # Should have 2 symbols after merge: [double_newline, single_newline]
    assert len(word.symbols) == 2
    assert word.symbols[0].c == double_newline_token
    assert word.symbols[0].len == 2  # Combined length of two newlines
    assert word.symbols[1].c == newline_token
    assert word.symbols[1].len == 1  # Single newline remains
    
    # Check the changes
    expected_changes = [
        ((newline_token, newline_token), -1),  # Remove first pair of newlines
        ((newline_token, newline_token), -1),  # Remove first pair of newlines
        ((double_newline_token, newline_token), 1)  # Add new pair with remaining newline
    ]
    assert changes == expected_changes

def test_merge_multiple_occurrences():
    """Test merge when there are multiple occurrences of the same pair."""
    word = Word()
    # Create pattern: A B A B C
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    # Merge A and B into 256 - should only merge the first occurrence
    changes = word.merge(65, 66, 256)
    # initial: 65, 66, 65, 66, 67
    # after 1st merge: 256, 65, 66, 67
    # after 2nd merge: 256, 256, 67
    
    # Should have 4 symbols after merge: [AB, AB, C]
    assert len(word.symbols) == 3
    assert word.symbols[0].c == 256
    assert word.symbols[1].c == 256
    assert word.symbols[2].c == 67

def test_merge_at_beginning():
    """Test merge at the beginning of the word."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    changes = word.merge(65, 66, 256)
    
    # Check that there's no previous symbol reference in changes
    assert len(word.symbols) == 2
    assert word.symbols[0].c == 256
    assert word.symbols[1].c == 67
    

def test_merge_at_end():
    """Test merge at the end of the word."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    changes = word.merge(66, 67, 256)
    
    # Should have 2 symbols: [A, 256]
    assert len(word.symbols) == 2
    assert word.symbols[0].c == 65
    assert word.symbols[1].c == 256
    
    # Should have changes for previous symbol pairing
    expected_changes = [
        ((66, 67), -1),  # Remove B-C pair
        ((65, 66), -1),  # Remove A-B pair
        ((65, 256), 1),   # Add (A, BC)
    ]
    assert changes == expected_changes

def test_merge_middle_symbol():
    """Test merge in the middle of the word."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    word.add(68, 1)  # 'D'
    
    changes = word.merge(66, 67, 256)
    
    # Should have 3 symbols: [A, 256, D]
    assert len(word.symbols) == 3
    assert word.symbols[0].c == 65
    assert word.symbols[1].c == 256
    assert word.symbols[2].c == 68
    
    # Should have changes for both previous and next symbol pairings
    expected_changes = [
        ((66, 67), -1),  # Remove B-C pair
        ((65, 66), -1),  # Remove A-B pair
        ((65, 256), 1),  # Add A-merged pair
        ((67, 68), -1),  # Remove C-D pair
        ((256, 68), 1)   # Add merged-D pair
    ]
    assert changes == expected_changes

def test_merge_preserves_byte_lengths():
    """Test that merge correctly combines byte lengths."""
    word = Word()
    word.add(65, 2)  # Symbol with length 2
    word.add(66, 3)  # Symbol with length 3
    word.add(67, 1)  # Symbol with length 1
    
    changes = word.merge(65, 66, 256)
    
    # Merged symbol should have combined length
    assert word.symbols[0].c == 256
    assert word.symbols[0].len == 5  # 2 + 3
    assert word.symbols[1].c == 67
    assert word.symbols[1].len == 1


def test_merge_all_basic():
    """Test basic merge_all functionality with a simple merge."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    
    # Define merge map: (A, B) -> (rank=1, new_id=256)
    merge_map = {
        (65, 66): (1, 256)
    }
    
    word.merge_all(merge_map)
    
    # Should have 2 active symbols after merge: merged AB and C
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 2
    assert active_symbols[0].c == 256
    assert active_symbols[0].len == 2
    assert active_symbols[1].c == 67
    assert active_symbols[1].len == 1


def test_merge_all_multiple_merges():
    """Test merge_all with multiple different merges."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    word.add(68, 1)  # 'D'
    
    
    # Define merge map with different priorities
    merge_map = {
        (65, 66): (1, 256),  # A+B -> 256 (priority 1)
        (67, 68): (2, 257)   # C+D -> 257 (priority 2)
    }
    
    word.merge_all(merge_map)
    
    # Should have 2 active symbols: [256, 257]
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 2
    assert active_symbols[0].c == 256
    assert active_symbols[0].len == 2
    assert active_symbols[1].c == 257
    assert active_symbols[1].len == 2


def test_merge_all_priority_order():
    """Test that merge_all respects priority order (lower rank = higher priority)."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    
    # Define overlapping merges with different priorities
    merge_map = {
        (65, 66): (2, 256),  # A+B -> 256 (priority 2)
        (66, 67): (1, 257)   # B+C -> 257 (priority 1, should execute first)
    }
    
    word.merge_all(merge_map)
    
    # B+C should merge first (priority 1), preventing A+B merge
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 2
    assert active_symbols[0].c == 65  # A remains
    assert active_symbols[0].len == 1  # A length
    assert active_symbols[1].c == 257  # BC merged
    assert active_symbols[1].len == 2  # BC merged length


def test_merge_all_cascading_merges():
    """Test merge_all with cascading merges (merge creates new mergeable pair)."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    
    # Define merge map where A+B creates AB, then AB+C can merge
    merge_map = {
        (65, 66): (1, 256),    # A+B -> 256 (priority 1)
        (256, 67): (2, 257)    # AB+C -> 257 (priority 2)
    }
    
    word.merge_all(merge_map)
    
    # Should cascade: A+B -> AB, then AB+C -> ABC
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 1
    assert active_symbols[0].c == 257
    assert active_symbols[0].len == 3


def test_merge_all_repeated_pattern():
    """Test merge_all with repeated patterns."""
    word = Word()
    # Create pattern: A B A B A B
    for _ in range(3):
        word.add(65, 1)  # 'A'
        word.add(66, 1)  # 'B'
    
    
    merge_map = {
        (65, 66): (1, 256)  # A+B -> 256
    }
    
    word.merge_all(merge_map)
    
    # All A+B pairs should be merged
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 3
    assert all(s.c == 256 for s in active_symbols)
    assert all(s.len == 2 for s in active_symbols)


def test_merge_all_empty_word():
    """Test merge_all with an empty word."""
    word = Word()
    merge_map = {(65, 66): (1, 256)}
    
    # Should not crash
    word.merge_all(merge_map)
    assert len(word.symbols) == 0


def test_merge_all_single_symbol():
    """Test merge_all with a single symbol (no pairs to merge)."""
    word = Word()
    word.add(65, 1)
    
    merge_map = {(65, 66): (1, 256)}
    
    word.merge_all(merge_map)
    
    # Single symbol should remain unchanged
    assert len(word.symbols) == 1
    assert word.symbols[0].c == 65


def test_merge_all_no_matching_pairs():
    """Test merge_all when no pairs match the merge map."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    
    
    # Merge map with non-existent pairs
    merge_map = {
        (88, 89): (1, 256),  # X+Y (not in word)
        (90, 91): (2, 257)   # Z+[ (not in word)
    }
    
    word.merge_all(merge_map)
    
    # Word should remain unchanged
    assert len(word.symbols) == 3
    assert word.symbols[0].c == 65
    assert word.symbols[1].c == 66
    assert word.symbols[2].c == 67


def test_merge_all_complex_cascading():
    """Test merge_all with complex cascading scenario."""
    word = Word()
    # Create: A B C D E
    for i in range(5):
        word.add(65 + i, 1)
    
    
    # Complex merge map with multiple levels
    merge_map = {
        (65, 66): (1, 256),    # A+B -> 256
        (67, 68): (2, 257),    # C+D -> 257
        (256, 67): (3, 258),   # AB+C -> 258 (but C will be merged with D first)
        (257, 69): (4, 259),   # CD+E -> 259
        (256, 257): (5, 260)   # AB+CD -> 260
    }
    
    word.merge_all(merge_map)
    
    # Expected: A+B->AB, C+D->CD, CD+E->CDE, AB+CDE remains separate
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 2
    # First should be AB (256)
    assert active_symbols[0].c == 256
    assert active_symbols[0].len == 2
    # Second should be CDE (259)
    assert active_symbols[1].c == 259
    assert active_symbols[1].len == 3


def test_merge_all_preserves_byte_lengths():
    """Test that merge_all correctly preserves and combines byte lengths."""
    word = Word()
    word.add(65, 2)  # 'A' with 2 bytes
    word.add(66, 3)  # 'B' with 3 bytes
    word.add(67, 1)  # 'C' with 1 byte
    
    merge_map = {
        (65, 66): (1, 256),
        (256, 67): (2, 257)
    }
    
    word.merge_all(merge_map)
    
    # Should have one symbol with combined length
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 1
    assert active_symbols[0].c == 257
    assert active_symbols[0].len == 6  # 2 + 3 + 1


def test_merge_all_same_priority_position_tiebreak():
    """Test that merge_all uses position as tiebreaker for same priority."""
    word = Word()
    # Create: A B A B
    word.add(65, 1)  # 'A' at position 0
    word.add(66, 1)  # 'B' at position 1
    word.add(65, 1)  # 'A' at position 2
    word.add(66, 1)  # 'B' at position 3
    
    
    # Same priority for both A+B pairs
    merge_map = {
        (65, 66): (1, 256)  # Both pairs have same priority
    }
    
    word.merge_all(merge_map)
    
    # Both pairs should be merged (they don't overlap)
    active_symbols = [s for s in word.symbols if s.len > 0]
    assert len(active_symbols) == 2
    assert all(s.c == 256 for s in active_symbols)


def test_merge_all_updates_prev_next_pointers():
    """Test that merge_all correctly updates prev/next pointers."""
    word = Word()
    word.add(65, 1)  # 'A'
    word.add(66, 1)  # 'B'
    word.add(67, 1)  # 'C'
    word.add(68, 1)  # 'D'
    
    merge_map = {
        (66, 67): (1, 256)  # B+C -> 256
    }
    
    word.merge_all(merge_map)
    
    # Check that pointers are correctly updated
    # A should still point to B (at position 1)
    assert word.symbols[0].next == 1

    # BC (at position 1) should have correct pointers
    merged_symbol = word.symbols[1]
    assert merged_symbol.c == 256
    assert merged_symbol.prev == 0  # Points to A
    assert merged_symbol.next == 3   # Points to D

    # D should point back to BC
    assert word.symbols[3].prev == 1
