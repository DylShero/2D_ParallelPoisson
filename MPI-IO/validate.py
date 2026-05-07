import numpy as np
import struct

def read_matrix_bin(filename):
    with open(filename, 'rb') as f:
        #Read the single integer dimension
        dim_bytes = f.read(4)
        dim = struct.unpack('i', dim_bytes)[0]
        
        #Read the rest of the doubles
        data = np.fromfile(f, dtype=np.float64)
        
        #Reconstruct the 20x20 matrix from the block column format
        A = np.zeros((dim, dim))
        idx = 0
        block_size = 5
        num_blocks = dim // block_size
        
        for bc in range(num_blocks):        #Block Columns
            for br in range(num_blocks):    #Block Rows
                for i in range(block_size): #Rows inside block
                    for j in range(block_size): #Cols inside block
                        A[br*block_size + i, bc*block_size + j] = data[idx]
                        idx += 1
        return A

def read_vector_bin(filename):
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        x = np.fromfile(f, dtype=np.float64)
        return x

#Read Binary Data
A = read_matrix_bin('mat-d20-b5-p4.bin')
x = read_vector_bin('x-d20.txt.bin')

#Calculate the exact result
y_true = A.dot(x)

#Read the MPI Output
try:
    y_mpi = np.loadtxt('mpi_result.txt')
except FileNotFoundError:
    print("No MPI result found")
    exit()

#Comparison
print("Python Result (First 5):", y_true[:5])
print("MPI Result    (First 5):", y_mpi[:5])

max_error = np.max(np.abs(y_true - y_mpi))
print(f"\nMaximum Absolute Error: {max_error:.5e}")

if max_error < 1e-10:
    print("Results match within tolerance")
else:
    print("Results do not match.")