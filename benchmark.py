import torch
import time
import numpy as np
from typing import Tuple

def benchmark_torch_sampling(bsz: int, vocab_size: int, num_trials: int = 100) -> Tuple[float, float]:
    """
    Test sampling speed and return mean time and standard deviation
    
    Args:
        bsz: batch size
        vocab_size: vocabulary size
        num_trials: number of test trials
    
    Returns:
        (mean_time, std_time): mean time and standard deviation (seconds)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.randn(bsz, vocab_size, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        probs = torch.softmax(logits, dim=-1)
        torch.multinomial(probs, num_samples=1)
    
    # Formal testing
    print(f"Starting torch sampling test for {num_trials} trials...")
    times = []
    for i in range(num_trials):
        start_time = time.perf_counter()
        probs = torch.softmax(logits, dim=-1)
        torch.multinomial(probs, num_samples=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        # if (i + 1) % 20 == 0:
        #     print(f"Completed: {i + 1}/{num_trials}")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time

def benchmark_gumbel_sampling(bsz: int, vocab_size: int, num_trials: int = 100) -> Tuple[float, float]:
    """
    Test sampling speed and return mean time and standard deviation
    
    Args:
        bsz: batch size
        vocab_size: vocabulary size
        num_trials: number of test trials
    
    Returns:
        (mean_time, std_time): mean time and standard deviation (seconds)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.randn(bsz, vocab_size, device=device)
    
    noise = []
    noise.append(-torch.log(-torch.log(torch.rand_like(logits))))

    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        a = (noise[0] + logits).argmax(dim=-1)
    
    # Formal testing
    print(f"Starting gumbel noise test for {num_trials} trials...")
    times = []
    for i in range(num_trials):
        start_time = time.perf_counter()
        a = (noise[0] + logits).argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        # if (i + 1) % 20 == 0:
        #     print(f"Completed: {i + 1}/{num_trials}")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time

def main():
    # Configure test parameter combinations
    configs = [
        (32, 32000, "Small Scale"),
        (128, 50000, "Medium Scale"),
        (512, 100000, "Large Scale")
    ]
    
    device = "GPU"# if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device}\n")
    
    for bsz, vocab_size, scale in configs:
        print(f"\nTesting {scale} configuration (bsz={bsz}, vocab_size={vocab_size}):")
        mean_time, std_time = benchmark_torch_sampling(bsz, vocab_size)
        print(f"Average torch time: {mean_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        mean_time, std_time = benchmark_gumbel_sampling(bsz, vocab_size)
        print(f"Average gumbel time: {mean_time*1000:.3f} ms ± {std_time*1000:.3f} ms")

if __name__ == "__main__":
    main()
