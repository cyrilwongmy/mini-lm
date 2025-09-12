import pytest
from cs336_basics.pretokenization import BPETokenizer
from .common import FIXTURES_PATH


"""Test suite for the pre_tokenization method of BPETokenizer."""

def test_empty_string():
    """Test pre_tokenization with an empty string."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("")
    assert result == {}

def test_single_word():
    """Test pre_tokenization with a single word."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("hello")
    expected = {b"hello": 1}
    assert result == expected

def test_multiple_words():
    """Test pre_tokenization with multiple words."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("hello world")
    expected = {
        b"hello": 1,
        b" world": 1
    }
    assert result == expected

def test_repeated_words():
    """Test pre_tokenization with repeated words to verify counting."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("hello hello world hello")
    expected = {
        b"hello": 1,
        b" hello": 2,
        b" world": 1
    }
    assert result == expected

def test_dash_in_word():
    """Test pre_tokenization with a word containing a dash."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("some text that i'll pre-tokenize")
    expected = {
        b"some": 1,
        b" text": 1,
        b" that": 1,
        b" i": 1,
        b"'ll": 1,
        b" pre": 1,
        b"-": 1,
        b"tokenize": 1
    }
    assert result == expected

def test_contractions():
    """Test pre_tokenization with contractions that match the pattern."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("don't can't I'll we're")
    # Based on the regex pattern: '(?:[sdmt]|ll|ve|re)
    expected = {
        b"don": 1,
        b"'t": 2,
        b" can": 1,
        b" I": 1,
        b"'ll": 1,
        b" we": 1,
        b"'re": 1
    }
    assert result == expected

def test_numbers():
    """Test pre_tokenization with numbers."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("123 456")
    expected = {
        b"123": 1,
        b" 456": 1
    }
    assert result == expected

def test_punctuation():
    """Test pre_tokenization with punctuation."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("Hello, world!")
    expected = {
        b"Hello": 1,
        b",": 1,
        b" world": 1,
        b"!": 1
    }
    assert result == expected

def test_whitespace_handling():
    """Test pre_tokenization with various whitespace patterns."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("word  word\nword\tword")
    # The regex handles different whitespace patterns
    # \s+(?!\S) matches whitespace at end, \s+ matches other whitespace
    expected = {
      b'word': 3,
      b' ': 1,
      b' word': 1,
      b'\n': 1,
      b'\t': 1,
    }
    assert result == expected

def test_unicode_characters():
    """Test pre_tokenization with Unicode characters."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("café naïve")
    expected = {
        b"caf\xc3\xa9": 1,  # café in UTF-8
        b" na\xc3\xafve": 1  # naïve in UTF-8
    }
    assert result == expected

def test_mixed_content():
    """Test pre_tokenization with mixed content including letters, numbers, and symbols."""
    tokenizer = BPETokenizer()
    text = "Hello123 @user #tag $100"
    result = tokenizer.pre_tokenization(text)
    
    # Verify that we get the expected number of tokens and proper byte encoding
    assert isinstance(result, dict)
    assert all(isinstance(key, bytes) for key in result.keys())
    assert all(isinstance(value, int) for value in result.values())
    assert all(value > 0 for value in result.values())

def test_with_fixture_file():
    """Test pre_tokenization with content from a fixture file."""
    with open(FIXTURES_PATH / "address.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization(content)
    
    # Verify basic properties
    assert isinstance(result, dict)
    assert len(result) > 0
    assert all(isinstance(key, bytes) for key in result.keys())
    assert all(isinstance(value, int) and value > 0 for value in result.values())
    
    # Check that some expected words are present
    assert any(b"Four" in key or key == b"Four" for key in result.keys())
    assert any(b"score" in key or key == b"score" for key in result.keys())

def test_german_text():
    """Test pre_tokenization with German text from fixture."""
    with open(FIXTURES_PATH / "german.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization(content)
    
    # Verify basic properties
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Check for German-specific characters
    german_chars_found = any(b"\xc3" in key for key in result.keys())  # UTF-8 encoded umlauts
    assert german_chars_found or len([k for k in result.keys() if len(k) > 1]) > 5  # Either umlauts or reasonable tokenization

def test_newlines_and_special_chars():
    """Test pre_tokenization with newlines and special characters."""
    tokenizer = BPETokenizer()
    text = "Line\nLine\r\nLine\tTabbed"
    result = tokenizer.pre_tokenization(text)
    
    assert isinstance(result, dict)
    assert len(result) > 0
    # Should handle newlines and tabs as whitespace
    expected = {
      b'Line': 3,
      b'\n': 2,
      b'\r': 1,
      b'\t': 1,
      b'Tabbed': 1,
    }
    assert result == expected

def test_return_type_and_structure():
    """Test that pre_tokenization returns the correct type and structure."""
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization("test input")
    
    # Verify return type
    assert isinstance(result, dict)
    
    # Verify key types (should be bytes)
    for key in result.keys():
        assert isinstance(key, bytes)
    
    # Verify value types (should be positive integers)
    for value in result.values():
        assert isinstance(value, int)
        assert value > 0

def test_byte_encoding_consistency():
    """Test that the function properly encodes strings to UTF-8 bytes."""
    tokenizer = BPETokenizer()
    text = "test"
    result = tokenizer.pre_tokenization(text)
    
    # The word "test" should be encoded as UTF-8 bytes
    expected_bytes = "test".encode("utf-8")
    assert expected_bytes in result
    assert result[expected_bytes] == 1

def test_large_counts():
    """Test pre_tokenization with repeated tokens to verify counting logic."""
    # Create a string with many repetitions
    word = "token "
    repetitions = 100
    text = word * repetitions
    
    tokenizer = BPETokenizer()
    result = tokenizer.pre_tokenization(text)
    
    # Should have high counts for repeated tokens
    assert any(count >= repetitions // 2 for count in result.values())
