CC = mpicc
CFLAGS = -g -Wall
LDFLAGS = -lm

# The 2D version depends on the 1D decomposer and the 2D Jacobi logic
POISSOBJS = decomp1d.o jacobi2d.o

EXECS = poiss2d

all: $(EXECS)

poiss2d: main2d.o $(POISSOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main2d.o: main2d.c jacobi2d.h poisson2d.h decomp1d.h
	$(CC) $(CFLAGS) -c main2d.c

jacobi2d.o: jacobi2d.c jacobi2d.h poisson2d.h
	$(CC) $(CFLAGS) -c jacobi2d.c

decomp1d.o: decomp1d.c decomp1d.h
	$(CC) $(CFLAGS) -c decomp1d.c

.PHONY: clean tags

clean:
	$(RM) *.o $(EXECS) TAGS tags global_solution.dat