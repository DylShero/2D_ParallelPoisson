#ifndef JACOBI2D_H
#define JACOBI2D_H

#include <mpi.h>

void sweep2d(double a[][maxn], double f[][maxn], int sx, int ex, int sy, int ey, double b[][maxn], int nx);

double griddiff2d(double a[][maxn], double b[][maxn], int sx, int ex, int sy, int ey);

void exchange2d_sendrecv(double a[][maxn], int sx, int ex, int sy, int ey, 
                         MPI_Comm cart_comm, int nbr_up, int nbr_down, int nbr_left, int nbr_right);

void exchange2d_nonblocking(double a[][maxn], int sx, int ex, int sy, int ey, 
                            MPI_Comm cart_comm, int nbr_up, int nbr_down, int nbr_left, int nbr_right);

#endif