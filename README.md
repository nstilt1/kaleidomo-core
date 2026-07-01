# Kaleidomo-core

Core crate that `kaleidomo` is based on for processing kaleidoscope videos and images in real-time on consumer hardware or in embedded environments.

## Features

* AVX2/SSE2/NEON SIMD optimizations for processing frames 2-4x faster than a software implementation running on the CPU.
* Multithreaded support + SIMD = comparable processing power of a GPU, but we have:
* WGSL support using `wgpu` crate that can target different graphics backends.

Overall, GPU processing could be very power efficient compared to MT SIMD, possibly being able to run on less than 1 GiB of system RAM if configured with an ultra-lightweight OS.
