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

void exchange2d_rma_fence(double a[][maxn], int sx, int ex, int sy, int ey, 
                          MPI_Win win, int nbr_up, int nbr_down, int nbr_left, int nbr_right) {
    MPI_Datatype col_type, row_type;
    
    //Top/Bottom boundaries 
    MPI_Type_contiguous(ey - sy + 1, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    
    //Left/Right boundaries 
    MPI_Type_vector(ex - sx + 1, 1, maxn, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    //Starting win fence
    MPI_Win_fence(0, win);

    //Send boundaries to the neighbors using MPI_Put
    
    if (nbr_up != MPI_PROC_NULL) {
        MPI_Aint target_disp = (MPI_Aint)(sx * maxn + sy);
        MPI_Put(&a[sx][sy], 1, row_type, nbr_up, target_disp, 1, row_type, win);
    }
    if (nbr_down != MPI_PROC_NULL) {
        MPI_Aint target_disp = (MPI_Aint)(ex * maxn + sy);
        MPI_Put(&a[ex][sy], 1, row_type, nbr_down, target_disp, 1, row_type, win);
    }
    if (nbr_left != MPI_PROC_NULL) {
        MPI_Aint target_disp = (MPI_Aint)(sx * maxn + sy);
        MPI_Put(&a[sx][sy], 1, col_type, nbr_left, target_disp, 1, col_type, win);
    }
    if (nbr_right != MPI_PROC_NULL) {
        MPI_Aint target_disp = (MPI_Aint)(sx * maxn + ey);
        MPI_Put(&a[sx][ey], 1, col_type, nbr_right, target_disp, 1, col_type, win);
    }

    //Close win fence
    MPI_Win_fence(0, win);

    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
}

void exchange2d_rma_pscw(double a[][maxn], int sx, int ex, int sy, int ey, 
                         MPI_Win win, MPI_Comm cart_comm, 
                         int nbr_up, int nbr_down, int nbr_left, int nbr_right) {
    MPI_Group world_group, nbr_group;
    MPI_Comm_group(cart_comm, &world_group);
    
    //Build a group containing only active neighbors
    int neighbors[4];
    int num_neighbors = 0;
    if (nbr_up != MPI_PROC_NULL) neighbors[num_neighbors++] = nbr_up;
    if (nbr_down != MPI_PROC_NULL) neighbors[num_neighbors++] = nbr_down;
    if (nbr_left != MPI_PROC_NULL) neighbors[num_neighbors++] = nbr_left;
    if (nbr_right != MPI_PROC_NULL) neighbors[num_neighbors++] = nbr_right;
    
    MPI_Group_incl(world_group, num_neighbors, neighbors, &nbr_group);

    MPI_Datatype col_type, row_type;
    MPI_Type_contiguous(ey - sy + 1, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(ex - sx + 1, 1, maxn, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    //Post Expose  buffers so neighbors can Put data
    MPI_Win_post(nbr_group, 0, win);

    MPI_Win_start(nbr_group, 0, win);

    if (nbr_up != MPI_PROC_NULL) {
        MPI_Put(&a[sx][sy], 1, row_type, nbr_up, (MPI_Aint)(sx * maxn + sy), 1, row_type, win);
    }
    if (nbr_down != MPI_PROC_NULL) {
        MPI_Put(&a[ex][sy], 1, row_type, nbr_down, (MPI_Aint)(ex * maxn + sy), 1, row_type, win);
    }
    if (nbr_left != MPI_PROC_NULL) {
        MPI_Put(&a[sx][sy], 1, col_type, nbr_left, (MPI_Aint)(sx * maxn + sy), 1, col_type, win);
    }
    if (nbr_right != MPI_PROC_NULL) {
        MPI_Put(&a[sx][ey], 1, col_type, nbr_right, (MPI_Aint)(sx * maxn + ey), 1, col_type, win);
    }

    //Complete to ensure all puts have finished
    MPI_Win_complete(win);
    
    MPI_Win_wait(win);

    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
    MPI_Group_free(&nbr_group);
    MPI_Group_free(&world_group);
}