#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobiseq.h"

#define maxit 1000

#include "decomp1d.h"

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn]);

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny);

void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);

int main(int argc, char **argv)
{
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int nx, ny;

  int it;
  double ldiff;
  /* double t1, t2; */
  double tol=1.0E-11;

  /* set the size of the problem */
  if(argc > 2){
    fprintf(stderr,"---->Usage: mpirun -np <nproc> %s <nx>\n",argv[0]);
    fprintf(stderr,"---->(for this code nx=ny)\n");
    exit(1);
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


  printf("nx = %d\n",nx);
  ny = nx;

  init_full_grids(a, b, f);

  printf("nx: %d\n",nx );

  onedinit_basic(a, b, f, nx, ny);
  print_full_grid(a);

  //  exit(20);

  ldiff = 1000;
  for(it=0; it<maxit; it++){

    sweep(a, f, nx, b);
    sweep(b, f, nx, a);

    ldiff = griddiffseq(a, b, nx);

    if(it%10==0){
      printf("diff beteween grids: %lf; \n",ldiff);
    }
    if( ldiff < tol ){
	printf("iterative solve converged\n");
      break;
    }

  }
    
  printf("DONE! (it: %d)\n",it);

  if( it == maxit ){
    fprintf(stderr,"Failed to converge\n");
  }
  /* printf("Run took %lf s\n",t2-t1); */

    print_grid_to_file("grid", a,  nx, ny);
    print_full_grid(a);

  return 0;
}

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny)
{
  int i,j;
  double left, bottom, right, top;

  left   = -1.0;
  bottom = 1.0;
  right  = 2.0;
  top    = 3.0;  

  /* set everything to 0 first */
  for(i=0; i<=nx+1; i++){
    for(j=0; j <= nx+1; j++){
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  /* deal with boundaries */
  for(i=1; i<=nx; i++){
    a[i][0] = bottom;
    b[i][0] = bottom;
    a[i][nx+1] = top;
    b[i][nx+1] = top;
  }

  for(j=1; j<nx+1; j++){
      a[0][j] = left;
      b[0][j] = left;
      a[nx+1][j] = right;
      b[nx+1][j] = right;
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
