"""
Utilities for encoding large text files with minimal memory usage.
"""

import os
import time
import numpy as np
from typing import Iterator, Tuple
from mini_lm.bpe.bpe_model import BpeModel


def read_file_in_chunks(file_path: str, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """
    Read a file in chunks, ensuring we don't split words at chunk boundaries.
    
    Args:
        file_path: Path to the text file
        chunk_size: Size of each chunk in bytes (default: 1MB)
        
    Yields:
        Text chunks that don't split words
    """
    with open(file_path, "r", encoding="utf-8") as f:
        remainder = ""
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if remainder:
                    yield remainder
                break
            
            # Prepend any remainder from the previous chunk
            chunk = remainder + chunk
            
            # If we're at the end of the file, yield everything
            if len(chunk) < chunk_size + len(remainder):
                yield chunk
                break
            
            # Find the last complete word boundary
            last_space = -1
            for i in range(len(chunk) - 1, len(chunk) - 1000, -1):  # Look back up to 1000 chars
                if chunk[i] in [' ', '\n', '\t']:
                    last_space = i
                    break
            
            if last_space > 0:
                # Yield up to the last space
                yield chunk[:last_space + 1]
                remainder = chunk[last_space + 1:]
            else:
                # No space found in the last 1000 chars, yield the whole chunk
                yield chunk
                remainder = ""


def encode_file_streaming(
    model: BpeModel, 
    data_path: str, 
    output_path: str,
    chunk_size: int = 1024 * 1024,
    buffer_size: int = 10_000_000
) -> dict:
    """
    Encode a large text file using streaming and save tokens incrementally.
    
    This implementation uses a single pass and writes tokens in batches,
    making it suitable for files larger than available RAM.
    
    Args:
        model: The BPE model to use for encoding
        data_path: Path to the input text file
        output_path: Path to save the encoded token IDs
        chunk_size: Size of each text chunk to process (default: 1MB)
        buffer_size: Number of tokens to buffer before writing (default: 10M)
        
    Returns:
        Dictionary with encoding statistics
    """
    print(f"Encoding {data_path} using streaming approach...")
    
    # Create a temporary file for incremental writes
    temp_output = output_path + ".tmp"
    
    # Statistics
    total_tokens = 0
    total_bytes = 0
    start_time = time.time()
    last_report_time = start_time
    last_report_bytes = 0
    
    # Token buffer
    token_buffer = []
    
    # Process file in chunks
    with open(temp_output, "wb") as out_file:
        for chunk_num, chunk in enumerate(read_file_in_chunks(data_path, chunk_size)):
            # Encode the chunk
            chunk_tokens = model.encode(chunk)
            token_buffer.extend(chunk_tokens)
            
            # Update statistics
            chunk_bytes = len(chunk.encode("utf-8"))
            total_bytes += chunk_bytes
            total_tokens += len(chunk_tokens)
            
            # Write buffer to disk if it's full
            if len(token_buffer) >= buffer_size:
                # Convert to numpy array and write
                buffer_array = np.array(token_buffer[:buffer_size], dtype=np.uint16)
                buffer_array.tofile(out_file)
                
                # Keep remaining tokens
                token_buffer = token_buffer[buffer_size:]
            
            # Progress report every 5 seconds or 100MB
            current_time = time.time()
            if (current_time - last_report_time >= 5.0 or 
                total_bytes - last_report_bytes >= 100 * 1024 * 1024):
                
                elapsed = current_time - start_time
                throughput = (total_bytes - last_report_bytes) / (current_time - last_report_time)
                overall_throughput = total_bytes / elapsed if elapsed > 0 else 0
                
                print(f"  Progress: {total_bytes / (1024**3):.2f} GB processed, "
                      f"{total_tokens:,} tokens, "
                      f"throughput: {throughput / (1024**2):.2f} MB/s, "
                      f"avg: {overall_throughput / (1024**2):.2f} MB/s")
                
                last_report_time = current_time
                last_report_bytes = total_bytes
        
        # Write any remaining tokens
        if token_buffer:
            buffer_array = np.array(token_buffer, dtype=np.uint16)
            buffer_array.tofile(out_file)
    
    # Convert the temporary file to a proper numpy file
    print(f"Converting to numpy format...")
    token_array = np.fromfile(temp_output, dtype=np.uint16)
    np.save(output_path, token_array)
    
    # Remove temporary file
    os.remove(temp_output)
    
    # Final statistics
    total_time = time.time() - start_time
    file_size = os.path.getsize(data_path)
    throughput = file_size / total_time if total_time > 0 else 0
    
    stats = {
        "file_path": data_path,
        "file_size_gb": file_size / (1024**3),
        "total_tokens": total_tokens,
        "total_time_seconds": total_time,
        "throughput_mb_per_sec": throughput / (1024**2),
        "compression_ratio": file_size / total_tokens if total_tokens > 0 else 0,
        "output_path": output_path,
        "output_size_mb": os.path.getsize(output_path) / (1024**2)
    }
    
    print(f"\nEncoding completed:")
    print(f"  File size: {stats['file_size_gb']:.2f} GB")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Total time: {stats['total_time_seconds']:.2f} seconds")
    print(f"  Average throughput: {stats['throughput_mb_per_sec']:.2f} MB/s")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f} bytes/token")
    print(f"  Output file: {output_path} ({stats['output_size_mb']:.2f} MB)")
    
    return stats


def estimate_memory_usage(file_size_gb: float, vocab_size: int = 32000) -> dict:
    """
    Estimate memory usage for encoding a file of given size.
    
    Args:
        file_size_gb: Size of the input file in GB
        vocab_size: Size of the tokenizer vocabulary
        
    Returns:
        Dictionary with memory estimates
    """
    # Rough estimates
    avg_compression_ratio = 3.5  # bytes per token (typical for English text)
    estimated_tokens = int(file_size_gb * 1024**3 / avg_compression_ratio)
    
    # Memory estimates
    two_pass_memory_gb = estimated_tokens * 2 / (1024**3)  # uint16 array
    streaming_memory_mb = 100  # Roughly constant for streaming approach
    
    return {
        "file_size_gb": file_size_gb,
        "estimated_tokens": estimated_tokens,
        "two_pass_memory_gb": two_pass_memory_gb,
        "streaming_memory_mb": streaming_memory_mb,
        "recommendation": "streaming" if file_size_gb > 4 else "two_pass"
    }


if __name__ == "__main__":
    # Example: Estimate memory usage for 12GB file
    estimates = estimate_memory_usage(12.0)
    print("Memory usage estimates for 12GB file:")
    print(f"  Estimated tokens: {estimates['estimated_tokens']:,}")
    print(f"  Two-pass approach memory: {estimates['two_pass_memory_gb']:.2f} GB")
    print(f"  Streaming approach memory: {estimates['streaming_memory_mb']:.0f} MB")
    print(f"  Recommended approach: {estimates['recommendation']}")