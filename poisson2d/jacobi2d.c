#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "poisson2d.h" 
#include "jacobi2d.h"

//2D Sweep only iterates over the local block
void sweep2d(double a[][maxn], double f[][maxn], int sx, int ex, int sy, int ey, double b[][maxn], int nx)
{
    double h = 1.0 / ((double)(nx + 1));
    for(int i = sx; i <= ex; i++){
        for(int j = sy; j <= ey; j++){
            b[i][j] = 0.25 * ( a[i-1][j] + a[i+1][j] + a[i][j+1] + a[i][j-1] - h*h*f[i][j] );
        }
    }
}

//2D Grid Diff
double griddiff2d(double a[][maxn], double b[][maxn], int sx, int ex, int sy, int ey)
{
    double sum = 0.0, tmp;
    for(int i = sx; i <= ex; i++){
        for(int j = sy; j <= ey; j++){
            tmp = (a[i][j] - b[i][j]);
            sum += tmp * tmp;
        }
    }
    return sum;
}

//Ghost Exchange (Sendrecv) using MPI_Type_vector for columns
void exchange2d_sendrecv(double a[][maxn], int sx, int ex, int sy, int ey, 
                         MPI_Comm cart_comm, int nbr_up, int nbr_down, int nbr_left, int nbr_right) 
{
    int num_rows = ex - sx + 1;
    int num_cols = ey - sy + 1;

    //Create a Vector datatype for the Left/Right columns (non-contiguous in memory)
    MPI_Datatype col_type;
    MPI_Type_vector(num_rows, 1, maxn, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    //Y-Direction (Left/Right)
    MPI_Sendrecv(&a[sx][ey], 1, col_type, nbr_right, 0, &a[sx][sy-1], 1, col_type, nbr_left, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&a[sx][sy], 1, col_type, nbr_left,  1, &a[sx][ey+1], 1, col_type, nbr_right, 1, cart_comm, MPI_STATUS_IGNORE);

    //X-Direction (Up/Down) - Contiguous rows
    MPI_Sendrecv(&a[ex][sy], num_cols, MPI_DOUBLE, nbr_down, 2, &a[sx-1][sy], num_cols, MPI_DOUBLE, nbr_up, 2, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&a[sx][sy], num_cols, MPI_DOUBLE, nbr_up,   3, &a[ex+1][sy], num_cols, MPI_DOUBLE, nbr_down, 3, cart_comm, MPI_STATUS_IGNORE);

    MPI_Type_free(&col_type);
}

//Ghost Exchange (Non-Blocking) 
void exchange2d_nonblocking(double a[][maxn], int sx, int ex, int sy, int ey, 
                            MPI_Comm cart_comm, int nbr_up, int nbr_down, int nbr_left, int nbr_right) 
{
    int num_rows = ex - sx + 1;
    int num_cols = ey - sy + 1;
    MPI_Request reqs[8];
    
    MPI_Datatype col_type;
    MPI_Type_vector(num_rows, 1, maxn, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    //Post Receives
    MPI_Irecv(&a[sx][sy-1], 1, col_type, nbr_left, 0, cart_comm, &reqs[0]);
    MPI_Irecv(&a[sx][ey+1], 1, col_type, nbr_right, 1, cart_comm, &reqs[1]);
    MPI_Irecv(&a[sx-1][sy], num_cols, MPI_DOUBLE, nbr_up, 2, cart_comm, &reqs[2]);
    MPI_Irecv(&a[ex+1][sy], num_cols, MPI_DOUBLE, nbr_down, 3, cart_comm, &reqs[3]);

    //Post Sends
    MPI_Isend(&a[sx][ey], 1, col_type, nbr_right, 0, cart_comm, &reqs[4]);
    MPI_Isend(&a[sx][sy], 1, col_type, nbr_left, 1, cart_comm, &reqs[5]);
    MPI_Isend(&a[ex][sy], num_cols, MPI_DOUBLE, nbr_down, 2, cart_comm, &reqs[6]);
    MPI_Isend(&a[sx][sy], num_cols, MPI_DOUBLE, nbr_up, 3, cart_comm, &reqs[7]);

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
    MPI_Type_free(&col_type);
}