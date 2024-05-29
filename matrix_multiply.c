#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> // Add this line for va_start and va_end
#include <mpi.h>
#include "cblas.h"
#include <sys/time.h>

#define SEED 42

// Define log level
#define LOG_LEVEL_DEBUG 1

// Log file name
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
        matrix[i] = (double)(rand() % size);
    }
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

// Perform matrix multiplication based on the bitstring
void multiply_matrices(int *bitstring, double *A0, double *A1, int m, int n, double *result) {
    double *temp_result = (double *)malloc(sizeof(double) * m * m);
    // Initialize result matrix with identity matrix
    for (int i = 0; i < m * m; i++) {
        result[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        double *matrix;
        if (bitstring[i] == 0) {
            matrix = A0;
        } else {
            matrix = A1;
        }
        // Perform matrix multiplication
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, matrix, m, result, m, 1.0, temp_result, m);
        // Copy result back to result matrix
        for (int j = 0; j < m * m; j++) {
            result[j] = temp_result[j];
        }
    }
    free(temp_result);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <m> <n> <size> <output_file>\n", argv[0]);
        exit(1);
    }

    int m = atoi(argv[1]); // Matrix size
    int n = atoi(argv[2]); // Bitstring length
    int size = atoi(argv[3]); // Max element size
    char *output_file = argv[4]; // Output file name

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
    multiply_matrices(bitstring, A0, A1, m, n, result);
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
