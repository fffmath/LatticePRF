#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <mpi.h>
#include "cblas.h"
#include <sys/time.h>

#define SEED 42
#define LOG_LEVEL_DEBUG 1
#define LOG_FILE "log.txt"

// Log function
void log_debug(const char *format, ...) {
    va_list args;
    va_start(args, format);
    FILE *log_file = fopen(LOG_FILE, "a");
    if (log_file != NULL) {
        vfprintf(log_file, format, args);
        fclose(log_file);
    }
    va_end(args);
}

// Generate a random matrix of size m x m with elements not exceeding size
void generate_random_matrix(double *matrix, int m, int size) {
    for (int i = 0; i < m * m; i++) {
        matrix[i] = (double)((rand() % (2 * size + 1)) - size);
    }

    #if LOG_LEVEL_DEBUG
    log_debug("Generated matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            log_debug("%lf ", matrix[i * m + j]);
        }
        log_debug("\n");
    }
    #endif
}

// Generate a random bitstring of length n
void generate_random_bitstring(int *bitstring, int n) {
    for (int i = 0; i < n; i++) {
        bitstring[i] = rand() % 2;
    }
    #if LOG_LEVEL_DEBUG
    log_debug("Generated bitstring: ");
    for (int i = 0; i < n; i++) {
        log_debug("%d ", bitstring[i]);
    }
    log_debug("\n");
    #endif
}

// Recursive function to perform matrix multiplication based on the bitstring
void multiply_matrices_recursive(int *bitstring, double **matrices, int start, int end, int m, double *result) {
    if (start == end) {
        // Base case: if start and end indices are the same, just copy the matrix
    double *matrix = matrices[bitstring[start]];
    for (int i = 0; i < m * m; i++) {
        result[i] = matrix[i];
    }
    } else {
        // Recursive case: split the range into two halves and apply matrix multiplication recursively
        int mid = (start + end) / 2;
        double *temp_result = (double *)malloc(sizeof(double) * m * m);

        // Left half multiplication
        multiply_matrices_recursive(bitstring, matrices, start, mid, m, result);
        // Right half multiplication
        multiply_matrices_recursive(bitstring, matrices, mid + 1, end, m, temp_result);

        // Multiply the results of left and right halves and add to the result
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, temp_result, m, result, m, 1.0, result, m);

        free(temp_result);
    }
}

// Wrapper function to initialize and call recursive multiplication
void multiply_matrices(int *bitstring, double **matrices, int n, int m, double *result) {
    // Initialize result matrix with identity matrix
    for (int i = 0; i < m * m; i++) {
        result[i] = 0.0;
    }
    // Call recursive multiplication function
    multiply_matrices_recursive(bitstring, matrices, 0, n - 1, m, result);
}

int main(int argc, char *argv[]) {
    int m = 128; // Default value for matrix size
    int size = 8; // Default maximum element size
    char *output_file = "output.txt"; // Default output file name

    // Check if the number of arguments is less than 2
    if (argc < 2) {
        printf("Usage: %s <n>\n", argv[0]); // Print usage message
        exit(1); // Exit the program with error status
    }

    int n = atoi(argv[1]); // Bitstring length from the first argument

    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get the rank and size of the communicator
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Seed the random number generator
    srand(SEED + rank * 12345); // Using rank as part of the seed

    // Generate random matrices A0 and A1
    double *A0 = (double *)malloc(sizeof(double) * m * m);
    double *A1 = (double *)malloc(sizeof(double) * m * m);
    generate_random_matrix(A0, m, size);
    generate_random_matrix(A1, m, size);

    // Generate random bitstring x
    int *bitstring = (int *)malloc(sizeof(int) * n);
    generate_random_bitstring(bitstring, n);

    // Allocate memory for result matrix
    double *result = (double *)malloc(sizeof(double) * m * m);

    // Perform matrix multiplication based on the bitstring
    double start_time = MPI_Wtime();
    double *matrices[] = {A0, A1}; // Array of matrices
    multiply_matrices(bitstring, matrices, n, m, result);
    double end_time = MPI_Wtime();

    // Write the result matrix to output file
    if (rank == 0) {
        FILE *fp = fopen(output_file, "w");
        if (fp == NULL) {
            printf("Error opening output file.\n");
            exit(1);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                fprintf(fp, "%lf ", result[i * m + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);

        printf("Output written to %s\n", output_file);
        printf("Execution time: %lf seconds\n", end_time - start_time);
    }

    // Clean up
    free(A0);
    free(A1);
    free(bitstring);
    free(result);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
