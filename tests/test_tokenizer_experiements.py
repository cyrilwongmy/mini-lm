"""
Test script to use pre-trained TinyStories and OpenWebText tokenizers
to encode validation data and calculate compression ratios.
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from mini_lm.bpe.bpe_model import BpeModel
from .encode_large_file_utils import encode_file_streaming, estimate_memory_usage

def load_tinystories_valid_tokenizer() -> BpeModel:
    """
    Loads the TinyStories tokenizer trained on the validation dataset.
    This tokenizer has a vocabulary of an unknown size.
    """
    vocab_path = "tests/TinyStoriesV2-GPT4-valid_vocab.json"
    merges_path = "tests/TinyStoriesV2-GPT4-valid_merges.txt"

    return BpeModel.from_file(
        vocab_path=vocab_path, merges_path=merges_path, special_tokens=["<|endoftext|>"]
    )

def load_tinystories_train_tokenizer() -> BpeModel:
    """
    Loads the TinyStories tokenizer trained on the training dataset.
    This tokenizer has a 10K vocabulary size.
    """
    vocab_path = "tests/TinyStoriesV2-GPT4-train_vocab.json"
    merges_path = "tests/TinyStoriesV2-GPT4-train_merges.txt"

    return BpeModel.from_file(
        vocab_path=vocab_path, merges_path=merges_path, special_tokens=["<|endoftext|>"]
    )

def load_owt_valid_tokenizer() -> BpeModel:
    """
    Loads the OpenWebText tokenizer trained on the validation dataset.
    This tokenizer has a vocabulary of an unknown size.
    """
    vocab_path = "tests/owt_valid_vocab.json"
    merges_path = "tests/owt_valid_merges.txt"

    return BpeModel.from_file(
        vocab_path=vocab_path, merges_path=merges_path, special_tokens=["<|endoftext|>"]
    )


def load_owt_train_tokenizer() -> BpeModel:
    """
    Loads the OpenWebText tokenizer trained on the training dataset.
    This tokenizer has a 32K vocabulary size.
    """
    vocab_path = "tests/owt_train_vocab.json"
    merges_path = "tests/owt_train_merges.txt"

    return BpeModel.from_file(
        vocab_path=vocab_path, merges_path=merges_path, special_tokens=["<|endoftext|>"]
    )


def encode_file_and_calculate_compression(
    tokenizer: BpeModel, file_path: str, tokenizer_name: str
):
    """
    Encodes a file using a tokenizer and calculates the compression ratio.
    Args:
        tokenizer: The BPE model to use for encoding.
        file_path: The path to the file to encode.
        tokenizer_name: The name of the tokenizer being used.
    Returns:
        A dictionary containing the results of the encoding.
    """
    print(f"\n=== {tokenizer_name} Tokenizer ===")

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Calculate the file size in bytes
    file_size_bytes = len(text.encode("utf-8"))

    # Encode the text into token IDs
    print(f"Encoding {file_path}...")
    token_ids = tokenizer.encode(text)

    # Calculate statistics
    num_tokens = len(token_ids)
    compression_ratio = file_size_bytes / num_tokens if num_tokens > 0 else 0

    # Print the results
    print(f"File: {file_path}")
    print(f"File size: {file_size_bytes:,} bytes")
    print(f"Number of tokens: {num_tokens:,}")
    print(f"Compression ratio: {compression_ratio:.2f} bytes/token")
    print(f"Vocabulary size: {len(tokenizer.vocab):,}")

    # Show the first few tokens as examples
    if num_tokens > 0:
        print(f"First 10 token IDs: {token_ids[:10]}")
        first_tokens_decoded = [tokenizer.decode([tid]) for tid in token_ids[:10]]
        print(f"First 10 tokens decoded: {first_tokens_decoded}")

    # Return the results as a dictionary
    return {
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "num_tokens": num_tokens,
        "compression_ratio": compression_ratio,
        "vocab_size": len(tokenizer.vocab),
        "tokenizer_name": tokenizer_name,
    }


def test_a():
    """
    Main function to run the tokenizer experiments A.
    Using previously-trained TinyStories and OpenWebText tokenizers
    (10K and 32K vocabulary size, respectively),
    encode these TinyStories and OpenWebText valid data into integer IDs.
    Compute each tokenizer’s compression ratio (bytes/token).
    """

    print("Loading tokenizers...")

    # Load tokenizers
    try:
        tinystories_tokenizer = load_tinystories_train_tokenizer()
        print(
            f"✓ TinyStories tokenizer loaded (vocab size: {len(tinystories_tokenizer.vocab)})"
        )
    except Exception as e:
        print(f"✗ Failed to load TinyStories tokenizer: {e}")
        return

    try:
        owt_tokenizer = load_owt_tokenizer()
        print(
            f"✓ OpenWebText tokenizer loaded (vocab size: {len(owt_tokenizer.vocab)})"
        )
    except Exception as e:
        print(f"✗ Failed to load OpenWebText tokenizer: {e}")
        return

    # A list to store the results of the encoding
    results = []

    # Encode the TinyStories validation file
    tinystories_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    if os.path.exists(tinystories_valid_path):
        result = encode_file_and_calculate_compression(
            tinystories_tokenizer, tinystories_valid_path, "TinyStories (10K vocab)"
        )
        results.append(result)
    else:
        print(f"✗ File not found: {tinystories_valid_path}")

    # Encode the OpenWebText validation file
    owt_valid_path = "data/owt_valid.txt"
    if os.path.exists(owt_valid_path):
        result = encode_file_and_calculate_compression(
            owt_tokenizer, owt_valid_path, "OpenWebText (32K vocab)"
        )
        results.append(result)
    else:
        print(f"✗ File not found: {owt_valid_path}")

    # Print a summary of the results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result['tokenizer_name']}:")
        print(f"  File: {result['file_path']}")
        print(f"  Vocabulary size: {result['vocab_size']:,}")
        print(f"  File size: {result['file_size_bytes']:,} bytes")
        print(f"  Number of tokens: {result['num_tokens']:,}")
        print(f"  Compression ratio: {result['compression_ratio']:.2f} bytes/token")

    # Compare the compression ratios of the two tokenizers
    assert len(results) == 2
    ts_ratio = results[0]["compression_ratio"]
    owt_ratio = results[1]["compression_ratio"]
    print(f"\nCompression Comparison:")
    print(f"  TinyStories: {ts_ratio:.2f} bytes/token")
    print(f"  OpenWebText: {owt_ratio:.2f} bytes/token")
    if ts_ratio > owt_ratio:
        print(f"  OpenWebText is {ts_ratio/owt_ratio:.2f}x more efficient")
    else:
        print(f"  TinyStories is {owt_ratio/ts_ratio:.2f}x more efficient")


def test_b():
    """
    Test B: Tokenize OpenWebText with TinyStories Tokenizer
    This experiment evaluates how well a tokenizer trained on a specific domain (TinyStories)
    compresses out-of-domain data (OpenWebText). This helps measure the tokenizer's generalization.
    """
    # Load the TinyStories tokenizer
    try:
        tinystories_tokenizer = load_tinystories_train_tokenizer()
        print(
            f"✓ TinyStories tokenizer loaded (vocab size: {len(tinystories_tokenizer.vocab)})"
        )
    except Exception as e:
        print(f"✗ Failed to load TinyStories tokenizer: {e}")
        return

    # Encode the OpenWebText validation data
    owt_valid_path = "data/owt_valid.txt"
    if os.path.exists(owt_valid_path):
        result = encode_file_and_calculate_compression(
            tinystories_tokenizer,
            owt_valid_path,
            "tinystories on OpenWebText (10K vocab)",
        )
    else:
        print(f"✗ File not found: {owt_valid_path}")

    assert (
        result is not None
    ), "Failed to get result for OpenWebText with TinyStories tokenizer"

    # Print a summary of the results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{result['tokenizer_name']}:")
    print(f"  File: {result['file_path']}")
    print(f"  Vocabulary size: {result['vocab_size']:,}")
    print(f"  File size: {result['file_size_bytes']:,} bytes")
    print(f"  Number of tokens: {result['num_tokens']:,}")
    print(f"  Compression ratio: {result['compression_ratio']:.2f} bytes/token")

    # Display the compression ratio
    ratio = result["compression_ratio"]
    print(f"\nCompression Comparison:")
    print(f"  TinyStories tokenizer on OpenWebText: {ratio:.2f} bytes/token")

def encode_file_and_serialize(model: BpeModel, data_path: str, output_path: str):
    """
    Encode a data file using the tokenizer and serialize the token IDs as a NumPy array.
    This function is suitable for files that can be loaded into memory.
    """
    # Read the data file into memory
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Encode the text and measure the time taken
    print(f"Encoding {data_path}...")
    time_start = time.time()
    token_ids = model.encode(text)
    time_end = time.time()
    print(f"Encoding completed in {time_end - time_start:.2f} seconds.")

    # Calculate and print the encoding throughput
    bytes = len(text.encode("utf-8"))
    throughput = bytes / (time_end - time_start) if (time_end - time_start) > 0 else 0
    print(f"Throughput: {throughput:.2f} bytes/sec")

    # Convert the token IDs to a NumPy array of uint16
    token_ids_array = np.array(token_ids, dtype=np.uint16)

    # Save the NumPy array to the output file
    np.save(output_path, token_ids_array)
    print(f"Token IDs saved to {output_path} (shape: {token_ids_array.shape}, dtype: {token_ids_array.dtype})")

def encode_iterable_and_serialize(
    model: BpeModel, data_path: str, output_path: str, chunk_size: int = 1024 * 1024
):
    """
    Encode a large text file iteratively and serialize the token IDs as a NumPy array.
    This function is designed to handle files that are too large to fit in memory. It uses a two-pass
    approach: the first pass counts the total number of tokens, and the second pass encodes the file
    and saves the token IDs to a NumPy array.
    Args:
        model: The BPE model to use for encoding.
        data_path: The path to the input text file.
        output_path: The path to save the encoded token IDs.
        chunk_size: The size of each chunk to process in bytes (default: 1MB).
    """
    print(f"Processing {data_path} iteratively...")

    # First pass: Count the total number of tokens to pre-allocate the array
    print("First pass: counting tokens...")
    total_tokens = 0
    bytes_processed = 0
    time_start = time.time()

    with open(data_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # To ensure accurate tokenization at chunk boundaries, read until a whitespace character
            # is found to avoid splitting words.
            if len(chunk) == chunk_size and chunk[-1] not in [" ", "\n", "\t"]:
                extra = ""
                while True:
                    char = f.read(1)
                    if not char or char in [" ", "\n", "\t"]:
                        break
                    extra += char
                chunk += extra

            token_ids = model.encode(chunk)
            total_tokens += len(token_ids)
            bytes_processed += len(chunk.encode("utf-8"))

            # Print a progress update every 100MB
            if bytes_processed % (100 * 1024 * 1024) == 0:
                elapsed = time.time() - time_start
                throughput = bytes_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  Processed {bytes_processed / (1024**3):.2f} GB, {total_tokens:,} tokens, "
                    f"{throughput / (1024**2):.2f} MB/s"
                )

    time_first_pass = time.time() - time_start
    print(f"First pass completed in {time_first_pass:.2f} seconds")
    print(f"Total tokens: {total_tokens:,}")

    # Allocate a NumPy array to store the token IDs
    print(f"Allocating array for {total_tokens:,} tokens...")
    token_array = np.zeros(total_tokens, dtype=np.uint16)

    # Second pass: Encode the file again and fill the pre-allocated array
    print("Second pass: encoding and saving tokens...")
    current_position = 0
    bytes_processed = 0
    time_start = time.time()

    with open(data_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Handle partial words at chunk boundaries, similar to the first pass
            if len(chunk) == chunk_size and chunk[-1] not in [" ", "\n", "\t"]:
                extra = ""
                while True:
                    char = f.read(1)
                    if not char or char in [" ", "\n", "\t"]:
                        break
                    extra += char
                chunk += extra

            token_ids = model.encode(chunk)

            # Copy the new token IDs to the array
            token_array[
                current_position : current_position + len(token_ids)
            ] = token_ids
            current_position += len(token_ids)
            bytes_processed += len(chunk.encode("utf-8"))

            # Print a progress update every 100MB
            if bytes_processed % (100 * 1024 * 1024) == 0:
                elapsed = time.time() - time_start
                throughput = bytes_processed / elapsed if elapsed > 0 else 0
                progress = (current_position / total_tokens) * 100
                print(
                    f"  Encoded {bytes_processed / (1024**3):.2f} GB, {progress:.1f}% complete, "
                    f"{throughput / (1024**2):.2f} MB/s"
                )

    time_second_pass = time.time() - time_start
    print(f"Second pass completed in {time_second_pass:.2f} seconds")

    # Save the final array to disk
    print(f"Saving to {output_path}...")
    np.save(output_path, token_array)

    # Print final statistics for the encoding process
    total_time = time_first_pass + time_second_pass
    file_size = os.path.getsize(data_path)
    throughput = file_size / total_time if total_time > 0 else 0

    print(f"\nEncoding completed:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  File size: {file_size / (1024**3):.2f} GB")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average throughput: {throughput / (1024**2):.2f} MB/s")
    print(f"  Compression ratio: {file_size / total_tokens:.2f} bytes/token")
    print(f"  Output saved to: {output_path}")


def encode_owt_valid():
    """
    Encode the OpenWebText validation data and serialize the token IDs to a NumPy array.
    This function uses a tokenizer specifically trained on the OpenWebText validation data.
    """
    # Load the OpenWebText validation tokenizer
    try:
        owt_tokenizer = load_owt_valid_tokenizer()
        print(
            f"✓ OpenWebText validation tokenizer loaded (vocab size: {len(owt_tokenizer.vocab)})"
        )
    except Exception as e:
        print(f"✗ Failed to load OpenWebText validation tokenizer: {e}")
        return

    # Define the input and output file paths
    data_path = "data/owt_valid.txt"
    output_path = "data/owt_valid_token_ids.npy"

    # Encode the file and save the token IDs
    encode_file_and_serialize(owt_tokenizer, data_path, output_path)


def encode_owt_train():
    """
    Encode the OpenWebText training data and serialize the token IDs to a NumPy array.
    This function uses a tokenizer trained on the OpenWebText training data and handles large
    files by choosing between a two-pass or streaming approach based on the file size.
    """
    # Load the OpenWebText training tokenizer
    try:
        owt_tokenizer = load_owt_train_tokenizer()
        print(f"✓ OpenWebText tokenizer loaded (vocab size: {len(owt_tokenizer.vocab)})")
    except Exception as e:
        print(f"✗ Failed to load OpenWebText tokenizer: {e}")
        return

    # Define the input and output file paths
    data_path = "data/owt_train.txt"
    output_path = "data/owt_train_token_ids.npy"

    # Check the file size to determine the best encoding method
    if os.path.exists(data_path):
        file_size_gb = os.path.getsize(data_path) / (1024 ** 3)
        print(f"File size: {file_size_gb:.2f} GB")

        # Estimate memory usage for both approaches
        estimates = estimate_memory_usage(file_size_gb)
        print(f"\nMemory usage estimates:")
        print(f"  Two-pass approach: ~{estimates['two_pass_memory_gb']:.1f} GB RAM")
        print(f"  Streaming approach: ~{estimates['streaming_memory_mb']:.0f} MB RAM")

        # Use a streaming approach for very large files to minimize memory usage
        if file_size_gb > 8:
            print(f"\nUsing streaming approach for large file...")
            encode_file_streaming(owt_tokenizer, data_path, output_path)
        else:
            print(f"\nUsing two-pass approach...")
            encode_iterable_and_serialize(owt_tokenizer, data_path, output_path)
    else:
        print(f"Error: File not found: {data_path}")


def encode_tinystories_valid():
    """
    Encode the TinyStories validation data and serialize the token IDs to a NumPy array.
    This function uses a tokenizer specifically trained on the TinyStories validation data.
    """
    # Load the TinyStories validation tokenizer
    try:
        tinystories_tokenizer = load_tinystories_valid_tokenizer()
        print(
            f"✓ TinyStories tokenizer loaded (vocab size: {len(tinystories_tokenizer.vocab)})"
        )
    except Exception as e:
        print(f"✗ Failed to load TinyStories tokenizer: {e}")
        return

    # Define the input and output file paths
    data_path = "data/TinyStoriesV2-GPT4-valid.txt"
    output_path = "data/TinyStories_valid_token_ids.npy"

    # Encode the file and save the token IDs
    encode_file_and_serialize(tinystories_tokenizer, data_path, output_path)


def encode_tinystories_train():
    """
    Encode the TinyStories training data and serialize the token IDs to a NumPy array.
    This function uses a tokenizer specifically trained on the TinyStories training data.
    """
    # Load the TinyStories training tokenizer
    try:
        tinystories_tokenizer = load_tinystories_train_tokenizer()
        print(
            f"✓ TinyStories tokenizer loaded (vocab size: {len(tinystories_tokenizer.vocab)})"
        )
    except Exception as e:
        print(f"✗ Failed to load TinyStories tokenizer: {e}")
        return

    # Define the input and output file paths
    data_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_path = "data/TinyStories_train_token_ids.npy"

    # Encode the file and save the token IDs
    encode_file_and_serialize(tinystories_tokenizer, data_path, output_path)


if __name__ == "__main__":
    # test_a()
    # test_b()
    # encode_owt_valid()
    encode_owt_train()
    # encode_tinystories_valid()
    # encode_tinystories_train()
