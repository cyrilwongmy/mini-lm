import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray | str, batch_size: int, context_length: int, device: str,
    mmap_mode: str | None = None, dtype: np.dtype = np.int64
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    The dataset can be either a regular numpy array or a memory-mapped array
    (created with np.memmap or loaded with np.load(..., mmap_mode='r')).
    Memory-mapped arrays are useful for datasets too large to fit in memory,
    as they load data on-demand from disk.

    Args:
        dataset (np.array, np.memmap, or str): 1D numpy array of integer token IDs in the dataset.
            Can be a regular array, a memory-mapped array, or a path to a file to load as memory-mapped.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
        mmap_mode (str or None): If dataset is a string path, this specifies the memory-map mode.
            Use 'r' for read-only memory-mapped access. Only used when dataset is a file path.
        dtype (np.dtype): Data type of the array when loading from file. Default is np.int64.
            Only used when dataset is a file path.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    
    Note:
        When using memory-mapped arrays, ensure the dtype matches the original saved array.
        You can verify the data looks correct by checking that all values are within
        the expected vocabulary size range.
    """
    # If dataset is a string path, load it as a memory-mapped array
    if isinstance(dataset, str):
        if mmap_mode is not None:
            # Load as memory-mapped array
            dataset = load_dataset_mmap(dataset, dtype=dtype)
        else:
            # Load as regular array
            dataset = np.load(dataset)
    
    # Calculate the maximum valid starting index
    # We need context_length tokens for input and 1 more for the last target token
    max_start_idx = len(dataset) - context_length
    
    # Sample random starting indices for each batch
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Create input sequences
    x = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, start_idx in enumerate(start_indices):
        x[i] = dataset[start_idx:start_idx + context_length]
    
    # Create targets (offset by 1)
    y = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, start_idx in enumerate(start_indices):
        y[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and place on the specified device
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    
    return x_tensor, y_tensor


def load_dataset_mmap(filepath: str, dtype: np.dtype = np.int64) -> np.memmap:
    """
    Load a dataset from disk as a memory-mapped array.
    
    This function is useful for loading large datasets that don't fit in memory.
    The data is loaded on-demand as it's accessed, rather than all at once.
    
    Args:
        filepath (str): Path to the numpy array file (.npy or .dat file).
        dtype (np.dtype): Data type of the array. Must match the original array's dtype.
            Default is np.int64 for token IDs.
    
    Returns:
        np.memmap: Memory-mapped array that can be used like a regular numpy array.
    
    Example:
        # For .npy files saved with np.save:
        dataset = np.load('data/tokens.npy', mmap_mode='r')
        
        # For raw binary files:
        dataset = load_dataset_mmap('data/tokens.dat', dtype=np.int64)
        
        # Verify the data looks correct
        print(f"Dataset shape: {dataset.shape}")
        print(f"Min token ID: {dataset.min()}, Max token ID: {dataset.max()}")
        print(f"First 10 tokens: {dataset[:10]}")
        
        # Use with get_batch
        x, y = get_batch(dataset, batch_size=32, context_length=128, device='cuda')
    """
    # For .npy files, use np.load with mmap_mode
    if filepath.endswith('.npy'):
        return np.load(filepath, mmap_mode='r')
    
    # For raw binary files, use np.memmap
    # Note: You need to know the shape beforehand for raw files
    # This assumes a 1D array; adjust shape as needed
    return np.memmap(filepath, dtype=dtype, mode='r')