# Lattice Based PRF

This program performs matrix multiplication based on a randomly generated bitstring using MPI for parallelization.

## Compilation

To compile the program, use the following command:

```bash
make
```

## Usage

To execute the program, run:

```bash
mpirun -np <num_processes> ./matrix_multiply <n>
```

Paras:
- <m>: Matrix size (default: 128)
- <n>: Bitstring length
- <size>: Max element size (default: 1024)
- <output_file>: Output file name (default: output.txt)

If you want to change the values of m, size, and output_file, you can do so in the source code.

### Author

You can find more information on [my personal website](https://www.fffmath.com/).

### License

This script is released under the MIT License 2.0. See the [LICENSE](LICENSE) file for details.
