import io
import os
import tempfile
import pytest
from cs336_basics.pretokenization import BPETokenizer


def test_empty_file():
    """Test find_chunk_boundaries with an empty file."""
    tokenizer = BPETokenizer()
    
    with io.BytesIO(b"") as f:
        chunks = tokenizer.find_chunk_boundaries(f, b"<|endoftext|>")
    
    # Should return [(0, 0)] for empty file
    assert chunks == [(0, 0)]


def test_single_chunk_small_file():
    """Test with a small file that should result in a single chunk."""
    tokenizer = BPETokenizer()
    content = b"Hello world this is a small file"
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, b"<|endoftext|>")
    
    # Should return [(0, file_size)] since no special tokens found
    expected = [(0, len(content))]
    assert chunks == expected


def test_file_with_single_special_token():
    """Test with a file containing a single special token."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = b"First chunk" + special_token + b"Second chunk"
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should create two chunks: before and after the special token
    token_pos = content.find(special_token)
    expected = [
        (0, token_pos),  # First chunk: from start to special token
        (token_pos + len(special_token), len(content))  # Second chunk: after special token to end
    ]
    assert chunks == expected


def test_file_with_multiple_special_tokens():
    """Test with a file containing multiple special tokens."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = b"First chunk" + special_token + b"Second chunk" + special_token + b"Third chunk"
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should create three chunks separated by special tokens
    first_token_pos = content.find(special_token)
    second_token_pos = content.find(special_token, first_token_pos + len(special_token))
    
    expected = [
        (0, first_token_pos),
        (first_token_pos + len(special_token), second_token_pos),
        (second_token_pos + len(special_token), len(content))
    ]
    assert chunks == expected


def test_special_token_at_beginning():
    """Test with special token at the very beginning of file."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = special_token + b"Content after token"
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should create one chunk after the initial special token
    expected = [(len(special_token), len(content))]
    assert chunks == expected


def test_special_token_at_end():
    """Test with special token at the very end of file."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = b"Content before token" + special_token
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should create one chunk before the final special token
    token_pos = content.find(special_token)
    expected = [(0, token_pos)]
    assert chunks == expected


def test_consecutive_special_tokens():
    """Test with consecutive special tokens."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = b"Start" + special_token + special_token + special_token + b"End"
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should create chunks around consecutive tokens
    first_pos = content.find(special_token)
    last_token_end = first_pos + 3 * len(special_token)  # After all three tokens
    
    expected = [
        (0, first_pos),  # Before first token
        (last_token_end, len(content))  # After last token
    ]
    assert chunks == expected


def test_only_special_tokens():
    """Test with a file containing only special tokens."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = special_token + special_token + special_token
    

    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should return empty list since there's no content between tokens
    expected = []
    assert chunks == expected


def test_large_file_simulation():
    """Test with a larger file that requires multiple chunk reads."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    
    # Create content larger than the internal chunk_size (8192)
    chunk_content = b"x" * 5000  # 5KB of content
    content = chunk_content + special_token + chunk_content + special_token + chunk_content
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    # Should find all special token positions correctly
    first_token_pos = len(chunk_content)
    second_token_pos = first_token_pos + len(special_token) + len(chunk_content)
    
    expected = [
        (0, first_token_pos),
        (first_token_pos + len(special_token), second_token_pos),
        (second_token_pos + len(special_token), len(content))
    ]
    assert chunks == expected


def test_special_token_spanning_chunk_boundary():
    """Test with special token that might span internal chunk boundaries."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    
    # Create content where special token is near the 8192 byte boundary
    content_before = b"x" * 8190  # Just before internal chunk boundary
    content_after = b"y" * 1000
    content = content_before + special_token + content_after
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    token_pos = content.find(special_token)
    expected = [
        (0, token_pos),
        (token_pos + len(special_token), len(content))
    ]
    assert chunks == expected


def test_different_special_tokens():
    """Test with different special token formats."""
    tokenizer = BPETokenizer()
    
    test_cases = [
        (b"<SEP>", b"text1<SEP>text2<SEP>text3"),
        (b"\n\n", b"paragraph1\n\nparagraph2\n\nparagraph3"),
        (b"|||", b"section1|||section2|||section3"),
        (b"END", b"startENDmiddleENDend")
    ]
    
    for special_token, content in test_cases:
        with io.BytesIO(content) as f:
            chunks = tokenizer.find_chunk_boundaries(f, special_token)
        
        # Verify chunks are properly separated
        positions = []
        start = 0
        while True:
            pos = content.find(special_token, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        expected_chunks = []
        start = 0
        for pos in positions:
            if pos > start:
                expected_chunks.append((start, pos))
            start = pos + len(special_token)
        
        if start < len(content):
            expected_chunks.append((start, len(content)))
        
        assert chunks == expected_chunks


def test_no_content_between_tokens():
    """Test edge case where there's no content between consecutive tokens."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = b"content" + special_token + special_token + b"more"
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    first_token_pos = content.find(special_token)
    second_token_end = first_token_pos + 2 * len(special_token)
    
    expected = [
        (0, first_token_pos),
        (second_token_end, len(content))
    ]
    assert chunks == expected


def test_special_token_assertion():
    """Test that non-bytes special_token raises assertion error."""
    tokenizer = BPETokenizer()
    content = b"Some content"
    
    with io.BytesIO(content) as f:
        with pytest.raises(AssertionError, match="Must represent special token as a bytestring"):
            tokenizer.find_chunk_boundaries(f, "<|endoftext|>")  # String instead of bytes


def test_real_file_operations():
    """Test with actual file operations using temporary files."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    content = b"First section" + special_token + b"Second section" + special_token + b"Third section"
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        
        try:
            with open(temp_file.name, 'rb') as f:
                chunks = tokenizer.find_chunk_boundaries(f, special_token)
            
            # Should find both special token positions
            first_token = content.find(special_token)
            second_token = content.find(special_token, first_token + len(special_token))
            
            expected = [
                (0, first_token),
                (first_token + len(special_token), second_token),
                (second_token + len(special_token), len(content))
            ]
            assert chunks == expected
            
        finally:
            os.unlink(temp_file.name)


def test_chunk_content_extraction():
    """Test that the returned chunks can be used to extract correct content."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    chunk1_content = b"This is the first chunk"
    chunk2_content = b"This is the second chunk"
    chunk3_content = b"This is the third chunk"
    
    content = chunk1_content + special_token + chunk2_content + special_token + chunk3_content
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
        
        # Extract content using the returned chunks
        extracted_chunks = []
        for start, end in chunks:
            f.seek(start)
            chunk_data = f.read(end - start)
            extracted_chunks.append(chunk_data)
    
    expected_chunks = [chunk1_content, chunk2_content, chunk3_content]
    assert extracted_chunks == expected_chunks


def test_unicode_content_with_special_tokens():
    """Test with unicode content and special tokens."""
    tokenizer = BPETokenizer()
    special_token = b"<|endoftext|>"
    
    # Create content with unicode characters
    unicode_text1 = "Hello 世界! 🌍"
    unicode_text2 = "Bonjour le monde! 🇫🇷"
    
    content = unicode_text1.encode('utf-8') + special_token + unicode_text2.encode('utf-8')
    
    with io.BytesIO(content) as f:
        chunks = tokenizer.find_chunk_boundaries(f, special_token)
    
    token_pos = content.find(special_token)
    expected = [
        (0, token_pos),
        (token_pos + len(special_token), len(content))
    ]
    assert chunks == expected
