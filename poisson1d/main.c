#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobi.h"

#define maxit 1000

#include "decomp1d.h"

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn]);

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e);

void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);

//New grid and boundary functions
double get_physical_coord(int index, int n);
double fzero(int xind, int yind, int nx, int ny, int s, int e);
double dbound_y0(int xind, int yind, int nx, int ny, int s, int e);
double ubound_y1(int xind, int yind, int nx, int ny, int s, int e);
double lbound_x0(int xind, int yind, int nx, int ny, int s, int e);
double rbound_x1(int xind, int yind, int nx, int ny, int s, int e);
double analytical_soln(int xind, int yind, int nx, int ny, int s, int e);
void onedinit_dirichlet(double a[][maxn], double b[][maxn], double f[][maxn],int nx, int ny, int s, int e,double (*lbound)(int, int, int, int, int, int),double 
                        (*dbound)(int, int, int, int, int, int),double (*rbound)(int, int, int, int, int, int),double (*ubound)(int, int, int, int, int, int));

void set_sub_grid_func(double u[][maxn], int nx, int ny, int s, int e,
                      double (*gf)(int xind, int yind, int nx, int ny, int s, int e));

double vinfnorm_diff_sub_grids(double u[][maxn], double v[][maxn],
                      int nx, int ny, int s, int e, MPI_Comm comm);

//Part 3 functions
void GatherGrid(double local_grid[][maxn], double global_grid[][maxn], 
                int nx, int ny, int s, int e, int myid, int nprocs, MPI_Comm comm);

void write_grid(char *fname, double grid[][maxn], int nx, int ny, int myid);            

int main(int argc, char **argv)
{
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int nx, ny;
  int myid, nprocs;
  /* MPI_Status status; */
  int nbrleft, nbrright, s, e, it;
  double glob_diff;
  double ldiff;
  double t1, t2;
  double tol=1.0E-11;

  //New MPI Cart Variables
  MPI_Comm cart_comm;
  int dims[1], periods[1];
  int reorder = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if( myid == 0 ){
    /* set the size of the problem */
    if(argc > 2){
      fprintf(stderr,"---->Usage: mpirun -np <nproc> %s <nx>\n",argv[0]);
      fprintf(stderr,"---->(for this code nx=ny)\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if( argc == 2 ){
      nx = atoi(argv[1]);
    }

    if( nx > maxn-2 ){
      fprintf(stderr,"grid size too large\n");
      exit(1);
    }
  }

  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("(myid: %d) nx = %d\n",myid,nx);
  ny = nx;

  init_full_grids(a, b, f);
  memset(f, 0, sizeof(f)); //Set grid to zeroS

  //Old neighbour finding
  /*nbrleft  = myid - 1;
  nbrright = myid + 1;

  if( myid == 0 ){
    nbrleft = MPI_PROC_NULL;
  }

  if( myid == nprocs-1 ){
    nbrright  = MPI_PROC_NULL;
  } */

  dims[0] = nprocs; //1D grid with 'nprocs' processes
  periods[0] = 0;   //0 means no wrap-around at boundaries

  //Create the Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);

  //Shift along dimension 0 with a displacement of 1 to find neighbors
  MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);

  MPE_Decomp1d(nx, nprocs, myid, &s, &e );

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n",myid, nx , s, e,
  	 nbrleft, nbrright);
  
  MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Abort(MPI_COMM_WORLD, 1); //From demonstration in class

  onedinit_dirichlet(a, b, f, nx, ny, s, e, lbound_x0, dbound_y0, rbound_x1, ubound_y1);

  t1 = MPI_Wtime();

  glob_diff = 1000;
  for(it=0; it<maxit; it++){

    exchang1(a, ny, s, e, cart_comm, nbrleft, nbrright); 
    sweep1d(a, f, nx, s, e, b);

    exchang1(b, ny, s, e, cart_comm, nbrleft, nbrright); 
    sweep1d(b, f, nx, s, e, a);

    glob_diff = vinfnorm_diff_sub_grids(a, b, nx, ny, s, e, cart_comm);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);//Not necessary to change all reduce as it should be all the same processes
    if(myid==0 && it%10==0){
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n",myid, ldiff, glob_diff);
    }
    if( glob_diff < tol ){
      if(myid==0){
  	printf("iterative solve converged\n");
      }
      break;
    }

  }
  
  t2=MPI_Wtime();
  
  printf("DONE! (it: %d)\n",it);

  if( myid == 0 ){
    if( it == maxit ){
      fprintf(stderr,"Failed to converge\n");
    }
    printf("Run took %lf s\n",t2-t1);
  }

  // Allocate memory for the global grid and exact solution
  double global_a[maxn][maxn];
  double exact_a[maxn][maxn];
  //Gather all local sub-grids into the complete global_a on Rank 0
  GatherGrid(a, global_a, nx, ny, s, e, myid, nprocs, MPI_COMM_WORLD);

  if (myid == 0) {
      //Rank 0 prints the complete grid to standard output
      printf("Gathered Global Grid Output\n");
      write_grid("global_solution.txt", global_a, nx, ny, myid);

      //Generate the exact analytical solution for the full grid
      set_sub_grid_func(exact_a, nx, ny, 1, nx, analytical_soln);
      
      //Compare the gathered solution to the exact solution
      double max_error = 0.0;
      for(int i = 1; i <= nx; i++){
          for(int j = 1; j <= ny; j++){
              double diff = fabs(global_a[i][j] - exact_a[i][j]);
              if(diff > max_error) {
                  max_error = diff;
              }
          }
      }

      printf("Max absolute error between Gathered Solution and Analytical Solution: %e\n", max_error);
  }


  MPI_Finalize();
  return 0;
}

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e)
{
  int i,j;


  double left, bottom, right, top;

  left   = -1.0;
  bottom = 1.0;
  right  = 2.0;
  top    = 3.0;  

  /* set everything to 0 first */
  for(i=s-1; i<=e+1; i++){
    for(j=0; j <= nx+1; j++){
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  /* deal with boundaries */
  for(i=s; i<=e; i++){
    a[i][0] = bottom;
    b[i][0] = bottom;
    a[i][nx+1] = top;
    b[i][nx+1] = top;
  }

  /* this is true for proc 0 */
  if( s == 1 ){
    for(j=1; j<nx+1; j++){
      a[0][j] = left;
      b[0][j] = left;
    }
  }
 
  /* this is true for proc size-1 */
  if( e == nx ){
    for(j=1; j<nx+1; j++){
      a[nx+1][j] = right;
      b[nx+1][j] = right;
    }

  }

}

void init_full_grid(double g[][maxn])
{
  int i,j;
  const double junkval = -5;

  for(i=0; i < maxn; i++){
    for(j=0; j<maxn; j++){
      g[i][j] = junkval;
    }
  }
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

/* prints to stdout in GRID view */
void print_full_grid(double x[][maxn])
{
  int i,j;
  for(j=maxn-1; j>=0; j--){
    for(i=0; i<maxn; i++){
      if(x[i][j] < 10000.0){
	printf("|%2.6lf| ",x[i][j]);
      } else {
	printf("%9.2lf ",x[i][j]);
      }
    }
    printf("\n");
  }

}

void print_in_order(double x[][maxn], MPI_Comm comm)
{
  int myid, size;
  int i;

  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);
  printf("Attempting to print in order\n");
  sleep(1);
  MPI_Barrier(comm);

  for(i=0; i<size; i++){
    if( i == myid ){
      printf("proc %d\n",myid);
      print_full_grid(x);
    }
    fflush(stdout);
    usleep(500);	
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny)
{
  FILE *fp;
  int i,j;

  fp = fopen(fname, "w");
  if( !fp ){
    fprintf(stderr, "Error: can't open file %s\n",fname);
    exit(4);
  }

  for(j=ny+1; j>=0; j--){
    for(i=0; i<nx+2; i++){
      fprintf(fp, "%lf ",x[i][j]);
      }
    fprintf(fp, "\n");
  }
  fclose(fp);
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

void onedinit_dirichlet(double a[][maxn], double b[][maxn], double f[][maxn],int nx, int ny, int s, int e,double (*lbound)(int, int, int, int, int, int),double 
                        (*dbound)(int, int, int, int, int, int),double (*rbound)(int, int, int, int, int, int),double (*ubound)(int, int, int, int, int, int))
                        {
  int i,j;

  //set everything to 0 first
  for(i=s-1; i<=e+1; i++){
    for(j=0; j <= ny+1; j++){
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  //deal with top and bottom boundaries
  for(i=s; i<=e; i++){
    a[i][0] = dbound(i, 0, nx, ny, s, e);
    b[i][0] = dbound(i, 0, nx, ny, s, e);
    a[i][ny+1] = ubound(i, ny+1, nx, ny, s, e); //ubound is at y-index ny+1
    b[i][ny+1] = ubound(i, ny+1, nx, ny, s, e);
  }

  //deal with left boundary 
  if( s == 1 ){
    for(j=0; j<ny+1; j++){
      a[0][j] = lbound(0, j, nx, ny, s, e);
      b[0][j] = lbound(0, j, nx, ny, s, e);
    }
  }
 
  //deal with right boundary 
  if( e == nx ){
    for(j=0; j<ny+1; j++){
      a[nx+1][j] = rbound(nx+1, j, nx, ny, s, e); //rbound is at x-index nx+1
      b[nx+1][j] = rbound(nx+1, j, nx, ny, s, e);
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

double vinfnorm_diff_sub_grids(double u[][maxn], double v[][maxn],
                              int nx, int ny, int s, int e, MPI_Comm comm)
{
  double diff, maxdiff;
  double ginorm = 0.0;
  int i,j;

  maxdiff = 0.0;
  for(i=s; i<e+1; i++){
    for(j=0; j<ny+2; j++){
      diff = fabs(u[i][j] - v[i][j]);
      if(diff > maxdiff){
        maxdiff = diff;
      }
    }
  }

  //Find the maximum difference across all processors 
  MPI_Allreduce(&maxdiff, &ginorm, 1, MPI_DOUBLE, MPI_MAX, comm);
    
  return ginorm;
}

void GatherGrid(double local_grid[][maxn], double global_grid[][maxn], 
                int nx, int ny, int s, int e, int myid, int nprocs, MPI_Comm comm) 
{
    //Calculate how many elements current process is responsible for
    int num_rows = e - s + 1;
    int num_elements = num_rows * maxn;

    //Rank 0 collects the data
    if (myid == 0) {
        
        //Rank 0 copies its own local data into the global grid
        for (int i = s; i <= e; i++) {
            for (int j = 0; j < maxn; j++) {
                global_grid[i][j] = local_grid[i][j];
            }
        }

        //Rank 0 receives data from all the other ranks (1 through nprocs-1)
        for (int p = 1; p < nprocs; p++) {
            int p_start, p_end;
            
            //Figure out which rows rank 'p' worked on
            MPE_Decomp1d(nx, nprocs, p, &p_start, &p_end);
            int p_elements = (p_end - p_start + 1) * maxn;

            //Receive rank p's data directly into the correct starting row of the global grid
            MPI_Recv(&global_grid[p_start][0], p_elements, MPI_DOUBLE, p, 0, comm, MPI_STATUS_IGNORE);
        }
    } 
    //All other ranks send their data to Rank 0
    else {
        MPI_Send(&local_grid[s][0], num_elements, MPI_DOUBLE, 0, 0, comm);
    }

    //Rank 0 re-applies the boundaries 
    if (myid == 0) {
        for(int i = 1; i <= nx; i++){
            global_grid[i][0]    = dbound_y0(i, 0, nx, ny, 1, nx);
            global_grid[i][ny+1] = ubound_y1(i, ny+1, nx, ny, 1, nx);
        }
        for(int j = 1; j <= ny; j++){
            global_grid[0][j]    = lbound_x0(0, j, nx, ny, 1, nx);
            global_grid[nx+1][j] = rbound_x1(nx+1, j, nx, ny, 1, nx);
        }
        
        //Corner points
        global_grid[0][0]       = 0.0;
        global_grid[nx+1][0]    = 0.0;
        global_grid[0][ny+1]    = ubound_y1(0, ny+1, nx, ny, 1, nx);
        global_grid[nx+1][ny+1] = ubound_y1(nx+1, ny+1, nx, ny, 1, nx);
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
