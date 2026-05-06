#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            printf("Must use 4 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_File fh_mat, fh_vec;
    MPI_Status status;
    int mat_dim, vec_len;
    int block_size = 5;

    
    //Open the matrix file
    MPI_File_open(MPI_COMM_WORLD, "mat-d20-b5-p4.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_mat);

    //Rank 0 reads the header (1 integer) and broadcasts it
    if (rank == 0) {
        MPI_File_read_at(fh_mat, 0, &mat_dim, 1, MPI_INT, &status);
    }
    MPI_Bcast(&mat_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Calculate elements per process
    int mat_elements_per_proc = mat_dim * block_size;
    double *local_mat = (double *)malloc(mat_elements_per_proc * sizeof(double));

    //Calculate offset
    MPI_Offset mat_offset = sizeof(int) + (rank * mat_elements_per_proc * sizeof(double));

    //Read the specific block column for this rank
    MPI_File_read_at_all(fh_mat, mat_offset, local_mat, mat_elements_per_proc, MPI_DOUBLE, &status);

    MPI_File_close(&fh_mat);



    // Open the vector binary file
    MPI_File_open(MPI_COMM_WORLD, "x-d20.txt.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_vec);

    //Rank 0 reads the header (1 integer) and broadcasts it
    if (rank == 0) {
        MPI_File_read_at(fh_vec, 0, &vec_len, 1, MPI_INT, &status);
    }
    MPI_Bcast(&vec_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *local_x = (double *)malloc(block_size * sizeof(double));

    //Calculate offset
    MPI_Offset vec_offset = sizeof(int) + (rank * block_size * sizeof(double));

    MPI_File_read_at_all(fh_vec, vec_offset, local_x, block_size, MPI_DOUBLE, &status);

    MPI_File_close(&fh_vec);

    if (rank == 0) {
        printf("Rank 0 sub-vector: %f, %f, %f, %f, %f\n", local_x[0], local_x[1], local_x[2], local_x[3], local_x[4]);
    }

    //clean up
    free(local_mat);
    free(local_x);
    MPI_Finalize();
    return 0;
}