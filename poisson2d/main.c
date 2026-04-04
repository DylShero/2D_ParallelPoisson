#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

#include "poisson2d.h"
#include "jacobi2d.h"
#include "decomp2d.h"

#define maxit 1000


void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn]);
double get_physical_coord(int index, int n);
double fzero(int xind, int yind, int nx, int ny, int s, int e);
double dbound_y0(int xind, int yind, int nx, int ny, int s, int e);
double ubound_y1(int xind, int yind, int nx, int ny, int s, int e);
double lbound_x0(int xind, int yind, int nx, int ny, int s, int e);
double rbound_x1(int xind, int yind, int nx, int ny, int s, int e);
double analytical_soln(int xind, int yind, int nx, int ny, int s, int e);

void onedinit_dirichlet_2d(double a[][maxn], double b[][maxn], double f[][maxn], int nx, int ny, int sx, int ex, int sy, int ey,
                        double (*lbound)(int, int, int, int, int, int), double (*dbound)(int, int, int, int, int, int),
                        double (*rbound)(int, int, int, int, int, int), double (*ubound)(int, int, int, int, int, int));

void set_sub_grid_func(double u[][maxn], int nx, int ny, int s, int e,
                       double (*gf)(int xind, int yind, int nx, int ny, int s, int e));

void GatherGrid2D(double local_grid[][maxn], double global_grid[][maxn], 
                  int nx, int ny, int sx, int ex, int sy, int ey, 
                  int myid, int nprocs, MPI_Comm cart_comm);

void write_grid(char *fname, double grid[][maxn], int nx, int ny, int myid);

int main(int argc, char **argv)
{
    double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
    int nx, ny, myid, nprocs, it;
    double glob_diff, ldiff, t1, t2;
    double tol = 1.0E-11;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int method = 1; //Default to SendRecv

    if( myid == 0 ){
        // Now accepts 2 arguments: <nx> and <method>
        if(argc < 2 || argc > 3){
            fprintf(stderr,"---->Usage: mpirun -np <nproc> %s <nx> [method: 1=SendRecv, 2=NonBlocking]\n",argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        nx = atoi(argv[1]);
        if (argc == 3) {
            method = atoi(argv[2]);
        }
        if( nx > maxn-2 ) {
            fprintf(stderr, "Grid size too large\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ny = nx;
    init_full_grids(a, b, f);

    //2D Cartesian Topology Setup
    MPI_Comm cart_comm;
    int dims[2] = {0, 0}; 
    int periods[2] = {0, 0}; 
    int coords[2];
    int nbr_up, nbr_down, nbr_left, nbr_right;

    //Automatically determine grid dimensions (e.g., 4 procs -> 2x2)
    MPI_Dims_create(nprocs, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, myid, 2, coords);

    //Find 4 neighbors for 2D ghost exchange
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down);   // Shift along X
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right); // Shift along Y

    //2D Decomposition
    int sx, ex, sy, ey;
    MPE_Decomp1d(nx, dims[0], coords[0], &sx, &ex);
    MPE_Decomp1d(ny, dims[1], coords[1], &sy, &ey);

    if (myid == 0) {printf("Running 2D Solver on %dx%d processor grid using %s method...\n", dims[0], dims[1], method == 1 ? "SendRecv" : "Non-Blocking");}

    //Initialize block and boundaries
    onedinit_dirichlet_2d(a, b, f, nx, ny, sx, ex, sy, ey, lbound_x0, dbound_y0, rbound_x1, ubound_y1);

    t1 = MPI_Wtime();
    glob_diff = 1000;
    
    //Iterative Solver Loop
    for(it = 0; it < maxit; it++){
        
        if (method == 1) {
            exchange2d_sendrecv(a, sx, ex, sy, ey, cart_comm, nbr_up, nbr_down, nbr_left, nbr_right); 
        } else {
            exchange2d_nonblocking(a, sx, ex, sy, ey, cart_comm, nbr_up, nbr_down, nbr_left, nbr_right);
        }
        sweep2d(a, f, sx, ex, sy, ey, b, nx);

        if (method == 1) {
            exchange2d_sendrecv(b, sx, ex, sy, ey, cart_comm, nbr_up, nbr_down, nbr_left, nbr_right); 
        } else {
            exchange2d_nonblocking(b, sx, ex, sy, ey, cart_comm, nbr_up, nbr_down, nbr_left, nbr_right);
        }
        sweep2d(b, f, sx, ex, sy, ey, a, nx);

        ldiff = griddiff2d(a, b, sx, ex, sy, ey);
        MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if(glob_diff < tol) break;
    }
    t2 = MPI_Wtime();

    if(myid == 0) printf("Converged in %d iterations. Time: %lf s\n", it, t2-t1);

    //Gather and Validate
    double global_a[maxn][maxn];
    double exact_a[maxn][maxn];

    GatherGrid2D(a, global_a, nx, ny, sx, ex, sy, ey, myid, nprocs, cart_comm);

    if (myid == 0) {
        write_grid("global_solution.txt", global_a, nx, ny, myid);
        
        //Validation 
        set_sub_grid_func(exact_a, nx, ny, 1, nx, analytical_soln);
        double max_error = 0.0;
        for(int i = 1; i <= nx; i++){
            for(int j = 1; j <= ny; j++){
                double diff = fabs(global_a[i][j] - exact_a[i][j]);
                if(diff > max_error) max_error = diff;
            }
        }
        printf("Validation: Max absolute error = %e\n", max_error);
    }

    MPI_Finalize();
    return 0;
}

/* set global a,b,f to initial arbitrarily chosen junk value */
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn])
{
  int i,j;
  const double junkval = -5;

  for(i=0; i < maxn; i++){
    for(j=0; j<maxn; j++){
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
    }
  }

}

double get_physical_coord(int index, int n) {
    return (double)index / (double)(n + 1);
}

//Forcing function f(x,y) = 0 
double fzero(int xind, int yind, int nx, int ny, int s, int e) {
    return 0.0;
}

//Bottom Boundary: u(x, 0) = 0 
double dbound_y0(int xind, int yind, int nx, int ny, int s, int e) {
    return 0.0;
}

//Top Boundary: u(x, 1) = 1 / ((1+x)^2 + 1) 
double ubound_y1(int xind, int yind, int nx, int ny, int s, int e) {
    double x = get_physical_coord(xind, nx);
    return 1.0 / (pow(1.0 + x, 2) + 1.0);
}

//Left Boundary: u(0, y) = y / (1 + y^2) 
double lbound_x0(int xind, int yind, int nx, int ny, int s, int e) {
    double y = get_physical_coord(yind, ny);
    return y / (1.0 + pow(y, 2));
}

//Right Boundary: u(1, y) = y / (4 + y^2) 
double rbound_x1(int xind, int yind, int nx, int ny, int s, int e) {
    double y = get_physical_coord(yind, ny);
    return y / (4.0 + pow(y, 2));
}

//Exact Analytical Solution for validation 
double analytical_soln(int xind, int yind, int nx, int ny, int s, int e) {
    double x = get_physical_coord(xind, nx);
    double y = get_physical_coord(yind, ny);
    return y / (pow(1.0 + x, 2) + pow(y, 2));
}

void GatherGrid2D(double local_grid[][maxn], double global_grid[][maxn], 
                  int nx, int ny, int sx, int ex, int sy, int ey, 
                  int myid, int nprocs, MPI_Comm cart_comm) 
{
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);

    int count = (ex - sx + 1) * (ey - sy + 1);
    double *temp_buf = (double*)malloc(count * sizeof(double));
    
    int idx = 0;
    for (int i = sx; i <= ex; i++) {
        for (int j = sy; j <= ey; j++) {
            temp_buf[idx++] = local_grid[i][j];
        }
    }

    if (myid == 0) {
        idx = 0;
        for (int i = sx; i <= ex; i++) {
            for (int j = sy; j <= ey; j++) {
                global_grid[i][j] = temp_buf[idx++];
            }
        }
        for (int p = 1; p < nprocs; p++) {
            int p_coords[2], p_sx, p_ex, p_sy, p_ey;
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            MPE_Decomp1d(nx, dims[0], p_coords[0], &p_sx, &p_ex);
            MPE_Decomp1d(ny, dims[1], p_coords[1], &p_sy, &p_ey);

            int p_count = (p_ex - p_sx + 1) * (p_ey - p_sy + 1);
            double *recv_buf = (double*)malloc(p_count * sizeof(double));

            MPI_Recv(recv_buf, p_count, MPI_DOUBLE, p, 0, cart_comm, MPI_STATUS_IGNORE);

            int p_idx = 0;
            for (int i = p_sx; i <= p_ex; i++) {
                for (int j = p_sy; j <= p_ey; j++) {
                    global_grid[i][j] = recv_buf[p_idx++];
                }
            }
            free(recv_buf);
        }
        
        //Boundaries
        for(int i = 1; i <= nx; i++){
            global_grid[i][0] = dbound_y0(i, 0, nx, ny, 1, nx);
            global_grid[i][ny+1] = ubound_y1(i, ny+1, nx, ny, 1, nx);
        }
        for(int j = 1; j <= ny; j++){
            global_grid[0][j] = lbound_x0(0, j, nx, ny, 1, nx);
            global_grid[nx+1][j] = rbound_x1(nx+1, j, nx, ny, 1, nx);
        }
    } else {
        MPI_Send(temp_buf, count, MPI_DOUBLE, 0, 0, cart_comm);
    }
    free(temp_buf);
}

void onedinit_dirichlet_2d(double a[][maxn], double b[][maxn], double f[][maxn],int nx, int ny, int sx, int ex, int sy, int ey,
                        double (*lbound)(int, int, int, int, int, int), double (*dbound)(int, int, int, int, int, int),
                        double (*rbound)(int, int, int, int, int, int), double (*ubound)(int, int, int, int, int, int))
{
    //Zero out local block
    for(int i = sx-1; i <= ex+1; i++){
        for(int j = sy-1; j <= ey+1; j++){
            a[i][j] = 0.0; b[i][j] = 0.0; f[i][j] = 0.0;
        }
    }
    //Top / Bottom
    if (sy == 1) { //Bottom edge
        for(int i = sx; i <= ex; i++){
            a[i][0] = dbound(i, 0, nx, ny, sx, ex);
            b[i][0] = dbound(i, 0, nx, ny, sx, ex);
        }
    }
    if (ey == ny) { //Top edge
        for(int i = sx; i <= ex; i++){
            a[i][ny+1] = ubound(i, ny+1, nx, ny, sx, ex);
            b[i][ny+1] = ubound(i, ny+1, nx, ny, sx, ex);
        }
    }
    //Left / Right
    if (sx == 1) { //Left edge
        for(int j = sy; j <= ey; j++){
            a[0][j] = lbound(0, j, nx, ny, sx, ex);
            b[0][j] = lbound(0, j, nx, ny, sx, ex);
        }
    }
    if (ex == nx) { //Right edge
        for(int j = sy; j <= ey; j++){
            a[nx+1][j] = rbound(nx+1, j, nx, ny, sx, ex);
            b[nx+1][j] = rbound(nx+1, j, nx, ny, sx, ex);
        }
    }
}

//Set a grid based on a mathematical function (used for exact analytical solution)
void set_sub_grid_func(double u[][maxn], int nx, int ny, int s, int e,
               double (*gf)(int xind, int yind, int nx, int ny, int s, int e))
{
  int i,j;
  for(i=s-1;i<=e+1; i++){
    for(j=0;j<ny+2;j++){
      u[i][j] = gf(i, j, nx, ny, s, e);
    }
  }
}

void write_grid(char *fname, double grid[][maxn], int nx, int ny, int myid)
{
    //Open file to make it easier to create heatmap after
    FILE *fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "(myid: %d) Error: can't open file %s\n", myid, fname);
        return;
    }

    //Print top-to-bottom, left-to-right 
    for (int j = ny + 1; j >= 0; j--) {
        for (int i = 0; i <= nx + 1; i++) {
            fprintf(fp, "%lf ", grid[i][j]);
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    printf("(myid: %d) Successfully wrote global grid to %s\n", myid, fname);
}