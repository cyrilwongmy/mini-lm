#!/usr/bin/env python3
"""
Sample script for decoding from a language model with configurable parameters.

This script demonstrates:
1. Generate completions for a user-provided prompt
2. Control the maximum number of generated tokens
3. Apply softmax temperature scaling
4. Top-p (nucleus) sampling
5. Load model from checkpoint (.pt format)
"""

import torch
import sys
from pathlib import Path

# Add mini_lm to Python path if running from project root
sys.path.append(str(Path(__file__).parent))

from mini_lm.nn import Transformer, decode
from mini_lm.bpe import BpeModel
from mini_lm.nn.optimizer import AdamW

# ===== CONFIGURATION PARAMETERS =====
# Model architecture parameters - adjust these to match your trained model
VOCAB_SIZE = 10000  
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0

# File paths
CHECKPOINT_PATH = "checkpoints/tinystories_512d_4L_16H_20M_1003_5PM/checkpoint_final.pt"  # Path to your model checkpoint
VOCAB_PATH = "data/TinyStoriesV2-GPT4-train_vocab.json"  # Path to vocabulary file
MERGES_PATH = "data/TinyStoriesV2-GPT4-train_merges.txt"  # Path to merges file

# Generation parameters
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0

# Special tokens
SPECIAL_TOKENS = ["<|endoftext|>"]


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, vocab_size: int = None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on (cuda/cpu)
        vocab_size: Vocabulary size (if None, uses global VOCAB_SIZE)
    
    Returns:
        Loaded transformer model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Use provided vocab_size or fall back to global
    model_vocab_size = vocab_size if vocab_size is not None else VOCAB_SIZE
    
    # Create model with configured architecture
    model = Transformer(
        vocab_size=model_vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Standard checkpoint format with optimizer state
        state_dict = checkpoint['model_state_dict']
        iteration = checkpoint.get('iteration', 0)
        print(f"Loaded checkpoint from iteration {iteration}")
    else:
        # Direct state dict
        state_dict = checkpoint
        print("Loaded model state dict")
    
    # Handle state dict with "_orig_mod." prefix (from torch.compile or DDP)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        # Remove the "_orig_mod." prefix from all keys
        state_dict = {key.replace('_orig_mod.', ''): value
                      for key, value in state_dict.items()}
        print("Removed '_orig_mod.' prefix from state dict keys")
    
    # Try to load the state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Error loading state dict with strict=True: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully with vocab_size={model_vocab_size}")
    return model


def load_tokenizer(vocab_path: str, merges_path: str):
    """
    Load BPE tokenizer from vocabulary and merges files.
    
    Args:
        vocab_path: Path to vocabulary JSON file
        merges_path: Path to merges text file
    
    Returns:
        BPE tokenizer model
    """
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    
    tokenizer = BpeModel.from_file(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=SPECIAL_TOKENS
    )
    
    # Update vocab size if needed
    global VOCAB_SIZE
    actual_vocab_size = len(tokenizer.vocab)
    if actual_vocab_size != VOCAB_SIZE:
        print(f"Updating vocab size from {VOCAB_SIZE} to {actual_vocab_size}")
        VOCAB_SIZE = actual_vocab_size
    
    return tokenizer


def generate_text(
    model: Transformer,
    tokenizer: BpeModel,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    device: torch.device = None,
    seed: int = None
):
    """
    Generate text completion for a given prompt.
    
    Args:
        model: The transformer language model
        tokenizer: BPE tokenizer for encoding/decoding
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for softmax scaling (higher = more random)
        top_p: Cumulative probability threshold for nucleus sampling
        device: Device to run generation on
        seed: Random seed for reproducible generation
    
    Returns:
        Tuple of (generated_tokens, generated_text)
    """
    if device is None:
        device = next(model.parameters()).device
    
    print(f"\nGenerating with parameters:")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    if seed is not None:
        print(f"  Seed: {seed}")
    
    # Generate completion
    tokens, text = decode(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        seed=seed,
    )
    
    return tokens, text


def main():
    """Main function to demonstrate text generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate text from a language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default parameters
  python sample.py "Once upon a time"
  
  # Generate with custom temperature
  python sample.py "The future of AI" --temperature 0.8
  
  # Generate with top-p sampling
  python sample.py "In the beginning" --top-p 0.9
  
  # Generate more tokens
  python sample.py "Hello world" --max-tokens 200
  
  # Use custom checkpoint
  python sample.py "Test prompt" --checkpoint path/to/model.pt
        """
    )
    
    # Required arguments
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    
    # Optional arguments
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                       help=f"Path to model checkpoint (default: {CHECKPOINT_PATH})")
    parser.add_argument("--vocab-path", type=str, default=VOCAB_PATH,
                       help=f"Path to vocabulary file (default: {VOCAB_PATH})")
    parser.add_argument("--merges-path", type=str, default=MERGES_PATH,
                       help=f"Path to merges file (default: {MERGES_PATH})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                       help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help=f"Temperature for sampling (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                       help=f"Top-p threshold for nucleus sampling (default: {DEFAULT_TOP_P})")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible generation")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                       help="Device to run on (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.temperature <= 0:
        parser.error("Temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("Top-p must be between 0 and 1")
    if args.max_tokens <= 0:
        parser.error("Max tokens must be positive")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(args.vocab_path, args.merges_path)
        
        # Get actual vocab size from tokenizer
        actual_vocab_size = len(tokenizer.vocab)
        
        # Load model with correct vocab size
        model = load_model_from_checkpoint(args.checkpoint, device, vocab_size=actual_vocab_size)
        
        # Generate text
        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)
        
        tokens, generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            seed=args.seed
        )
        
        print(f"\nGenerated text:")
        print(generated_text)
        print(f"\nTotal tokens generated: {len(tokens)}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("\nPlease ensure the following files exist:")
        print(f"  - Checkpoint: {args.checkpoint}")
        print(f"  - Vocabulary: {args.vocab_path}")
        print(f"  - Merges: {args.merges_path}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())