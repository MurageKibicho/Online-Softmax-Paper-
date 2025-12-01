# Online-Softmax-Paper in CUDA and C
Unofficial C and CUDA [LeetArxiv](https://leetarxiv.substack.com/p/cuda-papers-day-1-online-softmax) implementation of the paper *Online Normalizer Calculation for Softmax* (Milakov & Gimelshein, 2018)

![Abstract](https://github.com/MurageKibicho/Online-Softmax-Paper-/blob/cf69c99bac8dc65e19247f1089b8d556ecbc83da/Online%20Softmax/Marketing%20Propaganda/Abstract.png)


Complete writeup and coding guide available [here](https://leetarxiv.substack.com/p/cuda-papers-day-1-online-softmax)

## Paper Summary
The 2018 paper Online Normalizer Calculation for Softmax (Milakov & Gimelshein, 2018) addresses two shortcomings with the original softmax:

1. The naive softmax suffers from underflow and overflow when inputs are extreme (Tianlong, 2025).

2. The safer version of the naive softmax cannot run in parallel on GPU (Wangkuiyi, 2025)

The authors use a pretty clever trick to calculate the online normalizer in one loop (Tianlong, 2025).

Instead of first finding the maximum, the authors propose rescaling the accumulated sum whenever a new max is encountered.

## Getting Started
You can run the Jupyter Notebook locally or online in this [Google Colab notebook](https://colab.research.google.com/drive/1erLSbhvkOcqL7RtgSJCkfhnzkZ-4yKDI#scrollTo=1uMaV7M6X8hz).

Follow the free writeup [here](https://leetarxiv.substack.com/p/cuda-papers-day-1-online-softmax)

The C version runs with
```
gcc Softmax.c -lm -o m.o && ./m.o
```

Feel free to reach out on Twitter [@murage_kibicho](https://x.com/murage_kibicho) or via kibicho.murage@gmail.com


