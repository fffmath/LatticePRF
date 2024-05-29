CC = mpicc
CFLAGS = -O3 -Wall
LIBS = -lm -lblas

all: matrix_multiply

matrix_multiply: matrix_multiply.c
	$(CC) $(CFLAGS) -o matrix_multiply matrix_multiply.c $(LIBS)

clean:
	rm -f matrix_multiply
