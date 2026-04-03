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
void  print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);

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
    if( argc == 1 ){
      nx=15;
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

  nbrleft  = myid - 1;
  nbrright = myid + 1;

  if( myid == 0 ){
    nbrleft = MPI_PROC_NULL;
  }

  if( myid == nprocs-1 ){
    nbrright  = MPI_PROC_NULL;
  }

  MPE_Decomp1d(nx, nprocs, myid, &s, &e );

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n",myid, nx , s, e,
	 nbrleft, nbrright);

  /* onedinit_fancy(a, b, f, nx, ny, s, e, fiserles2, fzero, fiserles3, fiserles1); */
  /* onedinit_fancy(a, b, f, nx, ny, s, e, fzero, fone, ftwo, fthree); */

  onedinit_basic(a, b, f, nx, ny, s, e);

  t1 = MPI_Wtime();

  glob_diff = 1000;
  for(it=0; it<maxit; it++){
    if( it == 0 ){
      printf("\n======> NB VERSION\n\n");
    }

    nbxchange_and_sweep(a, f, nx, ny, s, e, b, MPI_COMM_WORLD, nbrleft, nbrright);

    /* printf("a AFWER\n");fflush(stdout);usleep(500); */
    /* MPI_Barrier(MPI_COMM_WORLD); */
    /* print_in_order(a, MPI_COMM_WORLD); */
    
    /* update a using b */
    nbxchange_and_sweep(b, f, nx, ny, s, e, a, MPI_COMM_WORLD, nbrleft, nbrright);

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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

  print_in_order(a, MPI_COMM_WORLD);
  if( nprocs == 1  ){
    print_grid_to_file("gridnb", a,  nx, ny);
    print_full_grid(a);
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
