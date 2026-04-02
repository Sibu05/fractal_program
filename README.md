# PC_Assignment-Fractals - Julia Set Fractal Parallelisation

## Overview
This project parallelises the generation of Julia Set fractals using OpenMP.
Five different parallelisation strategies are implemented and compared.

## Files
- `fractal.cpp` - Main source file containing all implementations
- `Makefile` - Build file
- `run.sh` - Script to run the program

## Implementations
1. **1D Rowwise** - Cyclic row distribution across threads
2. **1D Columnwise** - Cyclic column distribution across threads
3. **2D Row-Block** - Contiguous block of rows per thread
4. **2D Column-Block** - Contiguous block of columns per thread
5. **OpenMP for** - Automatic parallelisation using `#pragma omp parallel for`

## Requirements
- g++ with OpenMP support
- Linux/WSL

## Compilation
```bash
make
```

## Running
```bash
./run.sh
```

## Output
The program prints speedup results for each method across thread counts
{1, 2, 4, 6, 8, 10, 12, 14, 16} and saves fractal images to `output/`.
