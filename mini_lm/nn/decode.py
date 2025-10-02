import torch
from torch import Tensor
from typing import Optional, Union, List
from jaxtyping import Float, Int
import numpy as np

from .transformer import Transformer


def decode(
    model: Transformer,
    prompt: Union[str, List[int], Int[Tensor, "seq_len"]],
    tokenizer: Optional['BpeModel'] = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> tuple[List[int], str]:
    """Generate text completions from a language model.
    
    This function generates completions for a given prompt by sampling from the model's
    predicted next-token distributions. It supports temperature scaling and top-p (nucleus)
    sampling for controlling the randomness and quality of generations.
    
    Args:
        model: The transformer language model to use for generation.
        prompt: The input prompt, either as:
            - A string (requires tokenizer)
            - A list of token IDs
            - A tensor of token IDs
        tokenizer: Optional BPE tokenizer for encoding string prompts and decoding outputs.
                  Required if prompt is a string.
        max_tokens: Maximum number of tokens to generate (default: 100).
        temperature: Temperature for softmax scaling. Higher values (>1) increase randomness,
                    lower values (<1) make the model more confident (default: 1.0).
        top_p: Cumulative probability threshold for nucleus sampling. Only tokens with
               cumulative probability up to top_p are considered (default: 1.0 = no filtering).
        eos_token_id: Token ID that signals end of generation. If None and tokenizer is provided,
                     will try to use tokenizer's <|endoftext|> token.
        device: Device to run generation on. If None, uses model's device.
        seed: Random seed for reproducible generation (optional).
    
    Returns:
        tuple[List[int], str]: A tuple containing:
            - List of generated token IDs (including the prompt)
            - Decoded string (if tokenizer is provided, otherwise empty string)
    
    Examples:
        >>> # Generate with string prompt
        >>> tokens, text = decode(model, "Once upon a time", tokenizer, max_tokens=50)
        
        >>> # Generate with token IDs
        >>> tokens, _ = decode(model, [1, 2, 3, 4], max_tokens=20, temperature=0.8)
        
        >>> # Generate with top-p sampling
        >>> tokens, text = decode(model, "The weather is", tokenizer, top_p=0.9)
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Convert prompt to tensor
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("Tokenizer is required when prompt is a string")
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    elif isinstance(prompt, list):
        input_ids = torch.tensor(prompt, dtype=torch.long, device=device)
    elif isinstance(prompt, torch.Tensor):
        input_ids = prompt.to(device=device, dtype=torch.long)
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")
    
    # Handle batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
    
    # Get EOS token ID
    if eos_token_id is None and tokenizer is not None:
        # Try to get <|endoftext|> token from tokenizer
        eos_bytes = b"<|endoftext|>"
        if hasattr(tokenizer, 'word_to_id') and eos_bytes in tokenizer.word_to_id:
            eos_token_id = tokenizer.word_to_id[eos_bytes]
    
    # Put model in eval mode
    model.eval()
    
    # Generate tokens
    generated_tokens = input_ids.squeeze(0).tolist()  # Start with prompt tokens
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            logits = model(input_ids, return_logits=True)  # [batch_size, seq_len, vocab_size]
            
            # Get logits for the last position
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find the smallest set of tokens with cumulative probability >= top_p
                # We keep tokens where the cumulative probability (excluding current token) is < top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask to the right to keep the first token above the threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                # Set probabilities of tokens to remove to 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0
                
                # Renormalize
                probs = probs / probs.sum()
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens
            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)
            
            # Check for EOS token
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            
            # Append to input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode to text if tokenizer is provided
    generated_text = ""
    if tokenizer is not None:
        generated_text = tokenizer.decode(generated_tokens)
    
    return generated_tokens, generated_text


def generate_batch(
    model: Transformer,
    prompts: List[Union[str, List[int]]],
    tokenizer: Optional['BpeModel'] = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = 0,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> tuple[List[List[int]], List[str]]:
    """Generate text completions for multiple prompts in a batch.
    
    This function is similar to decode() but processes multiple prompts efficiently
    in a single batch. It handles variable-length prompts by padding.
    
    Args:
        model: The transformer language model to use for generation.
        prompts: List of prompts, each can be either a string or list of token IDs.
        tokenizer: Optional BPE tokenizer for encoding/decoding strings.
        max_tokens: Maximum number of tokens to generate per prompt.
        temperature: Temperature for softmax scaling.
        top_p: Cumulative probability threshold for nucleus sampling.
        eos_token_id: Token ID that signals end of generation.
        pad_token_id: Token ID to use for padding shorter sequences (default: 0).
        device: Device to run generation on.
        seed: Random seed for reproducible generation.
    
    Returns:
        tuple[List[List[int]], List[str]]: A tuple containing:
            - List of generated token ID lists (one per prompt)
            - List of decoded strings (if tokenizer provided)
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Process each prompt individually for now
    # (Full batch processing would require handling variable-length sequences)
    all_tokens = []
    all_texts = []
    
    for prompt in prompts:
        tokens, text = decode(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            device=device,
            seed=None,  # Don't reset seed for each prompt
        )
        all_tokens.append(tokens)
        all_texts.append(text)
    
    return all_tokens, all_texts


def beam_search_decode(
    model: Transformer,
    prompt: Union[str, List[int], Int[Tensor, "seq_len"]],
    tokenizer: Optional['BpeModel'] = None,
    max_tokens: int = 100,
    beam_size: int = 4,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[List[int], str]:
    """Generate text using beam search decoding.
    
    This is an alternative to sampling that maintains multiple hypotheses
    and returns the most likely sequence.
    
    Args:
        model: The transformer language model.
        prompt: Input prompt (string, token list, or tensor).
        tokenizer: Optional tokenizer for string encoding/decoding.
        max_tokens: Maximum tokens to generate.
        beam_size: Number of beams to maintain.
        temperature: Temperature for softmax scaling.
        eos_token_id: End-of-sequence token ID.
        device: Device to run on.
    
    Returns:
        tuple[List[int], str]: Best sequence tokens and decoded text.
    """
    # This is a placeholder for beam search implementation
    # For now, we'll use regular sampling with low temperature
    return decode(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=min(temperature, 0.1),  # Use low temperature for more deterministic output
        top_p=1.0,
        eos_token_id=eos_token_id,
        device=device,
    )