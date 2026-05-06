CC = mpicc
CFLAGS = -g -Wall -O3
LDFLAGS = -lm

EXEC = mpi_io_test

all: $(EXEC)

$(EXEC): main.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

.PHONY: clean

clean:
	$(RM) *.o $(EXEC)