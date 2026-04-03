#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobiseq.h"


void sweep(double a[][maxn], double f[][maxn], int nx, double b[][maxn])
{
  double h;
  int i,j;

  h = 1.0/((double)(nx+1));

  for(i=1; i<nx+1; i++){
    for(j=1; j<nx+1; j++){
      b[i][j] = 0.25 * ( a[i-1][j] + a[i+1][j] + a[i][j+1] + a[i][j-1]  - h*h*f[i][j] );
    }
  }
}

double griddiffseq(double a[][maxn], double b[][maxn], int nx)
{
  double sum;
  double tmp;
  int i, j;

  sum = 0.0;

  for(i=1; i<nx+1; i++){
    for(j=1;j<nx+1;j++){
      tmp = (a[i][j] - b[i][j]);
      sum = sum + tmp*tmp;
    }
  }

  return sum;

}
