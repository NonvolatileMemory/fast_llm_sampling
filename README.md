# Fast Multinomial Sampling with Gumbel-Max

**TL;DR**: We present a significant performance optimization for PyTorch's multinomial sampling using the Gumbel-Max trick, achieving up to 2.8x speedup on A100 GPUs.

## Overview

This project implements an efficient alternative to `torch.multinomial` using the Gumbel-Max trick for sampling from categorical distributions. Our approach shows substantial performance improvements across various batch sizes and vocabulary sizes, particularly beneficial for large-scale language model inference.

## Performance Benchmarks

Benchmarks conducted on NVIDIA A100 80GB GPU:

### Small Scale (batch_size=32, vocab_size=32000)
- **Torch Multinomial**: 0.600 ms ± 0.058 ms
- **Gumbel-Max**: 0.214 ms ± 0.004 ms
- **Speedup**: 2.8x

### Medium Scale (batch_size=128, vocab_size=50000)
- **Torch Multinomial**: 4.549 ms ± 2.609 ms
- **Gumbel-Max**: 1.294 ms ± 0.009 ms
- **Speedup**: 3.5x

### Large Scale (batch_size=512, vocab_size=100000)
- **Torch Multinomial**: 64.386 ms ± 2.748 ms
- **Gumbel-Max**: 30.544 ms ± 1.725 ms
- **Speedup**: 2.1x


## Implementation

The Gumbel-Max trick converts sampling from a categorical distribution into an argmax operation over Gumbel noise, which can be highly optimized on modern GPUs:

```python
def gumbel_sample(logits):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.argmax(logits + gumbel_noise, dim=-1)
```
