import json
import time
import tracemalloc
from pathlib import Path

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

from mini_lm.bpe.bpe_trainer import BpeTrainer
from mini_lm.bpe.bpe_model import BpeModel


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    print("starting test_train_bpe_special_tokens")
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
    print("ending test_train_bpe_special_tokens")

def train_bpe_and_output_summary(training_data_path: str, vocab_size: int, special_token: str | None = None):
    print("Starting training BPE tokenizer using ", training_data_path)

    training_data_name = training_data_path.split('.')[0]

    # Set up file paths
    data_dir = Path(__file__).parent.parent / "data"
    input_path = data_dir / training_data_path
    output_dir = Path(__file__).parent
    
    # Output file paths
    training_summary_output_path = output_dir / f"{training_data_name}_training_summary.txt"

    # Start memory tracking
    tracemalloc.start()
    
    # Start time tracking
    start_time = time.time()
    
    # Train BPE tokenizer
    print(f"Training BPE on {input_path} with vocab_size={vocab_size}...")
    tokenizer = BpeTrainer()
    bpe_model = tokenizer.do_train(input_path, vocab_size=vocab_size, special_tokens=[special_token] if special_token else None)
    print(f"Training complete. Vocabulary size: {len(bpe_model.vocab)}, Merges: {len(bpe_model.merges)}")

    # End time tracking
    end_time = time.time()
    training_time_seconds = end_time - start_time
    training_time_hours = training_time_seconds / 3600
    
    # Get peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
    tracemalloc.stop()

    print(f"Finding longest token...")
    # Find the longest token in the vocabulary
    longest_token = b""
    longest_token_id = -1
    for token_id, token_bytes in bpe_model.vocab.items():
        if len(token_bytes) > len(longest_token):
            longest_token = token_bytes
            longest_token_id = token_id


    # Save vocab and merges
    bpe_model.save(folder=output_dir, name=training_data_name)
    
    # Write training summary to text file
    print(f"Saving training summary to {training_summary_output_path}")
    with open(training_summary_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Time: {training_time_hours:.2f} hours ({training_time_seconds:.2f} seconds)\n")
        f.write(f"Peak Memory Usage: {peak_memory_mb:.2f} MB\n")
        f.write(f"Longest Token: {longest_token.decode('utf-8', errors='ignore')} (length: {len(longest_token)} bytes)\n")
        f.write(f"Longest Token ID: {longest_token_id}\n")
        f.write(f"Total Vocabulary Size: {len(bpe_model.vocab)}\n")

    # Print results to console
    print(f"\nTraining completed successfully!")
    print(f"Training Time: {training_time_hours:.2f} hours ({training_time_seconds:.2f} seconds)")
    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
    print(f"Longest Token: '{longest_token.decode('utf-8', errors='ignore')}' (length: {len(longest_token)} bytes)")
    print(f"Total Vocabulary Size: {len(bpe_model.vocab)}")

    # Basic assertions to ensure the training worked correctly
    assert len(bpe_model.vocab) == 10000, f"Expected vocab size 10000, got {len(bpe_model.vocab)}"
    assert b"<|endoftext|>" in bpe_model.vocab.values(), "Special token <|endoftext|> not found in vocabulary"
    assert len(bpe_model.merges) > 0, "No merges were created"

    print("Ending training BPE tokenizer using ", training_data_path)

def test_train_bpe_tinystories_valid():
    train_bpe_and_output_summary("TinyStoriesV2-GPT4-valid.txt", vocab_size=10000, special_token="<|endoftext|>")

def test_train_bpe_tinystories():
    train_bpe_and_output_summary("TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_token="<|endoftext|>")

def test_train_bpe_owt_valid():
    train_bpe_and_output_summary("owt_valid.txt", vocab_size=10000, special_token="<|endoftext|>")

def test_train_bpe_owt_train():
    train_bpe_and_output_summary("owt_train.txt", vocab_size=10000, special_token="<|endoftext|>")