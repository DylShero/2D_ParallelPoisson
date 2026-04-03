#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobi.h"

#define maxit 20000

#include "decomp1d.h"

#ifdef _PROFMPE
#include "mpe_log.h"
#endif

void get_win_nbr_group(const MPI_Win win, const int nbrleft, const int nbrright, MPI_Group *nbr_group);

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn]);

void set_sub_grid_func(double u[][maxn], int nx, int ny, int s, int e,
		      double (*gf)(int xind, int yind, int nx, int ny, int s, int e));
void set_full_grid_func(double u[][maxn], int nx, int ny, int s, int e,
			double (*gf)(int , int , int , int , int , int ));

/* vector inf norm between u,v with grids */
double vinfnorm_diff_sub_grids(double u[][maxn], double v[][maxn],
			       int nx, int ny, int s, int e, MPI_Comm comm);

void  print_full_grid(double x[][maxn]);
void  print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);
void print_in_order(double x[][maxn], MPI_Comm comm);

void onedinit_fancy(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e,
		    double (*lbound)(int , int , int , int , int , int ),
		    double (*dbound)(int , int , int , int , int , int ),
		    double (*rbound)(int , int , int , int , int , int ),
		    double (*ubound)(int , int , int , int , int , int ));

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e);

void GatherGrid(double a[][maxn], int nx, int ny, MPI_Comm comm);
void GatherGridCopy(double g[][maxn], double a[][maxn], int nx, int ny, MPI_Comm comm);
void copy_full_grid(double a[][maxn], double b[][maxn]);

int main(int argc, char **argv)
{
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  double u[maxn][maxn];
  double ggrid[maxn][maxn];
  MPI_Win wina, winb;

  MPI_Group nbr_groupa, nbr_groupb;

  MPI_Aint right_ghost_disp, my_right_ghost_disp;
  double infnorm, ginorm;
  int nx, ny;
  int myid, nprocs;
  /* MPI_Status status; */
  int nbrleft, nbrright, s, e, it;
  double glob_diff;
  double ldiff;
  double t1, t2;
  double tol=1.0E-11;
  int putput = 0;

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

  /* printf("(myid: %d)Got this far\n", myid); */
  
  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("(myid: %d) nx = %d\n",myid,nx);
  ny = nx;
  if( myid == 0){
    printf("====> hx: %lf; hy: %lf\n",1.0/((double)(nx+1)),
	   1.0/((double)(ny+1)));

    printf("rma1\n");

    if( putput ){
      printf("\n======> using putput exchange <======\n");
    } else {
      printf("\n======> using getput exchange <======\n");
    }
  }

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

  /* Iserles example */
  /* onedinit_fancy(a, b, f, nx, ny, s, e, fiserles2, fzero, fiserles3, fiserles1); */

  /* Olver, p159 example */
  /* onedinit_fancy(a, b, f, nx, ny, s, e, fzero, olverp159lower, fzero, fzero); */
  /* onedinit_fancy(a, b, f, nx, ny, s, e, fzero, fone, ftwo, fthree); */

  onedinit_basic(a, b, f, nx, ny, s, e);

  /* print_in_order(a, MPI_COMM_WORLD); */
  
  t1 = MPI_Wtime();

  MPI_Win_create(&a[s-1][0], (maxn)*(e-s+3)*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &wina);
  MPI_Win_create(&b[s-1][0], (maxn)*(e-s+3)*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winb);

  get_win_nbr_group(wina, nbrleft, nbrright, &nbr_groupa);
  get_win_nbr_group(winb, nbrleft, nbrright, &nbr_groupb);

  if( putput ){
    my_right_ghost_disp = 1+(maxn)*(e-s+2);
    MPI_Sendrecv(&my_right_ghost_disp, 1, MPI_AINT, nbrright, 0,
		 &right_ghost_disp, 1, MPI_AINT, nbrleft, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  glob_diff = 1000;
  for(it=0; it<maxit; it++){

  MPI_Barrier(MPI_COMM_WORLD);
    /* if( myid == 0){ */
    /*   printf("BEFORE EXCHANGE(nbrleft: %d; nbrright: %d\n",nbrleft,nbrright); */
    /*   print_full_grid(a); */
    /* } */
  /* printf("(myid: %d) BEFORE EXCHANGE(nbrleft: %d; nbrright: %d; right_ghost_disp: %ld\n",myid, nbrleft,nbrright, right_ghost_disp); */

    MPI_Barrier(MPI_COMM_WORLD);

    if( putput ){
      exchangpscw1(a, ny, s, e, wina, nbrleft, nbrright, right_ghost_disp, nbr_groupa);

    } else {
      exchangpscw2(a, ny, s, e, wina, nbrleft, nbrright, nbr_groupa);
    }

    /* print_in_order(a, MPI_COMM_WORLD); */

    /* MPI_Barrier(MPI_COMM_WORLD); */
    /* if( myid == 1){ */
    /*   printf("AFTER EXCHANGE(nbrleft: %d; nbrright: %d\n",nbrleft,nbrright); */
    /*   print_full_grid(a); */
    /* } */
    sweep1d(a, f, nx, s, e, b);

    if( putput ){

      exchangpscw1(b, ny, s, e, winb, nbrleft, nbrright, right_ghost_disp, nbr_groupb);

    } else {
      exchangpscw2(b, ny, s, e, winb, nbrleft, nbrright, nbr_groupb);
    }
    sweep1d(b, f, nx, s, e, a);

    /* if( it > 1 ){ */
    /* MPI_Barrier(MPI_COMM_WORLD); */
    /* MPI_Finalize(); */
    /* return 0; */
    /* } */

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(myid==0 && it%100==0){
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n",myid, ldiff, glob_diff);
    }
    if( glob_diff < tol ){
      if(myid==0){
	printf("iterative solve converged\n");
      }
      break;
    }
  }
  printf("it: %d\n",it);


  MPI_Barrier(MPI_COMM_WORLD);
  t2=MPI_Wtime();
  if( nprocs == 1 ){
    print_grid_to_file("grid", a, nx, ny);
  }
  
  if( myid == 0 ){
    if( it == maxit ){
      fprintf(stderr,"Failed to converge\n");
    }
    printf("Run took %lf s\n",t2-t1);
  }

  /* printf("NUMERICAL GRID\n"); */
  /* set_sub_grid_func(u, nx,  ny, s,  e, fiserlessoln); */
  /* print_full_grid(a); */
  /* printf("ISERLES GRID\n"); */
  /* set_sub_grid_func(u, nx,  ny, s,  e, seriessoln1); */
  /* print_full_grid(u); */
  infnorm  =  vinfnorm_diff_sub_grids(u, b, nx, ny, s, e, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(&infnorm, &ginorm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if( myid==0 ){
    printf("Diff between analytial solution and 'a': %E \n",ginorm);
  }

  init_full_grid(ggrid);
  GatherGridCopy(ggrid, a, nx, ny, MPI_COMM_WORLD);

  if( myid == 0 ){
    print_grid_to_file("pargrid", ggrid,  nx, ny);
    print_full_grid(ggrid);
  }
  MPI_Finalize();

  return 0;
}

/* ranks must have size <max number of neighbours> (2 in 1D poisson) */
void get_win_nbr_group(const MPI_Win win, const int nbrleft, const int nbrright, MPI_Group *nbr_group)
{
  MPI_Group wingroup;
  int nbr_ranks[] = {0, 0};
  int n_in_group = 0;

  MPI_Win_get_group(win, &wingroup);
  if( nbrleft != MPI_PROC_NULL){
    nbr_ranks[n_in_group++] = nbrleft;
  }
  if( nbrright != MPI_PROC_NULL){
    nbr_ranks[n_in_group++] = nbrright;
  }
  MPI_Group_incl(wingroup, n_in_group, nbr_ranks, nbr_group);
  MPI_Group_free(&wingroup);
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

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e)
{
  int i,j;
  double left, bottom, right, top;
  int tmprank;

  MPI_Comm_rank(MPI_COMM_WORLD, &tmprank);

  left   = -1.0;
  bottom = 1.0;
  right  = 2.0;
  top    = 3.0;  

  /* set everything to 0 first in local grid */
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

/* initialise "local" array */
void onedinit_fancy(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e,
		    double (*lbound)(int xind, int yind, int nx, int ny, int s, int e),
		    double (*dbound)(int xind, int yind, int nx, int ny, int s, int e),
		    double (*rbound)(int xind, int yind, int nx, int ny, int s, int e),
		    double (*ubound)(int xind, int yind, int nx, int ny, int s, int e))
{
  int i,j;

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
    a[i][0] = dbound(i, 0, nx, ny, s, e);
    b[i][0] = dbound(i, 0, nx, ny, s, e);
    a[i][ny+1] = ubound(i, 0, nx, ny, s, e);
    b[i][ny+1] = ubound(i, 0, nx, ny, s, e);;
  }

  /* this is true for proc 0 */
 if( s == 1 ){
    for(j=1; j<nx+1; j++){
      a[0][j] = lbound(0, j, nx, ny, s, e);;
      b[0][j] = lbound(0, j, nx, ny, s, e);
    }
  }
 
   /* this is true for proc size-1 */
 if( e == nx ){
    for(j=1; j<nx+1; j++){
      a[nx+1][j] = rbound(0, j, nx, ny, s, e);;
      b[nx+1][j] = rbound(0, j, nx, ny, s, e);
    }

  }

}

/* set "my" part of the grid */
void set_sub_grid_func(double u[][maxn], int nx, int ny, int s, int e,
		       double (*gf)(int xind, int yind, int nx, int ny, int s, int e))
{
  int i,j;

  printf("ssgf: s %d, e %d\n",s,e);
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
  double ginorm=0.0;
  int maxi, maxj;
  int i,j;

  maxdiff = 0.0;
  maxi=-1; maxj=-1;
  for(i=s;i<e+1; i++){
    for(j=0;j<ny+2;j++){
      diff=fabs(u[i][j] - v[i][j]);
      if(diff > maxdiff){
	maxdiff = diff;
	maxi = i; maxj = j;
      }
    }
  }

  printf("maxi: %d; maxj: %d\n",maxi, maxj);
  printf("maxdiff: %E\n",maxdiff);

  MPI_Allreduce(&maxdiff, &ginorm, 1, MPI_DOUBLE, MPI_MAX, comm);
    
  return ginorm;

}

void GatherGrid(double a[][maxn], int nx, int ny, MPI_Comm comm)
{
  double **subgrid;
  double *tmp;
  int myid, nprocs;
  int *gridsizes;
  int s, e;
  int localcols;
  int subgrid_index;
  MPI_Status status;
  /* const int root = 0; */
  int i, j, k;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if( nprocs == 1)
    return;

  if( myid == 0 ){

    gridsizes = (int *)calloc(nprocs, sizeof(int));
    for(i=1; i < nprocs; ++i){
      MPE_Decomp1d(nx, nprocs, i, &s, &e );
      gridsizes[i] = (e - s + 1) + 2;
      
      subgrid = (double **)malloc(gridsizes[i] * sizeof(double *));
      tmp    = (double *)malloc(gridsizes[i]*(nx+2)*sizeof(double));
      for(j=0; j < gridsizes[i]; ++j){
	subgrid[j] = &tmp[j*(nx+2)];
      }

      /* for(k=0; k < gridsizes[i]; ++k){ */
      /* 	for(j=0; j < (nx+2); ++j){ */
      /* 	  printf("--%lf ",subgrid[k][j]); */
      /* 	} */
      /* 	printf("\n"); */
      /* } */

      MPI_Recv(tmp, gridsizes[i]*(nx+2), MPI_DOUBLE, i, i, comm, &status);

      for(k=s; k < e+2; ++k){
      	for(j=0; j < (nx+2); ++j){
	  a[k][j] = subgrid[k - s + 1][j];
      	}
      }

      free(subgrid);
      free(tmp);
    }
    free(gridsizes);

  } else {
    MPE_Decomp1d(nx, nprocs, myid, &s, &e );
    localcols =  (e - s + 1) + 2;
    subgrid = (double **)malloc(localcols * sizeof(double *));
    tmp    = (double *)malloc(localcols*(nx+2)*sizeof(double));
    for(i=0; i < localcols; ++i){
      subgrid[i] = &tmp[i*(nx+2)];
    }
    for(i=s-1; i<e+2; ++i){
      subgrid_index = i - s + 1;
      if( subgrid_index < 0){
	fprintf(stderr,"==============> Error: index cannot be negative: %d\n",subgrid_index);
	exit(6);
      }
      for(j=0; j < nx+2; ++j){
	subgrid[subgrid_index][j] = a[i][j];
      }
    }

    MPI_Send(tmp, localcols * (nx+2), MPI_DOUBLE, 0, myid, comm);

    free(subgrid);
    free(tmp);
  }

}

void GatherGridCopy(double g[][maxn], double a[][maxn], int nx, int ny, MPI_Comm comm)
{
  double **subgrid;
  double *tmp;
  int myid, nprocs;
  int *gridsizes;
  int s, e;
  int localcols;
  int subgrid_index;
  MPI_Status status;
  /* const int root = 0; */
  int i, j, k;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if( nprocs == 1){
    copy_full_grid(g, a);
    return;
  }

  if( myid == 0 ){

    MPE_Decomp1d(nx, nprocs, myid, &s, &e );
    /* copy rank 0 part of a into g */
    for(i=s-1; i < e+2; ++i){
      for(j=0; j < (nx+2); ++j){
	g[i][j] = a[i][j];
      }
    }

    /* recv and copy remote segments */
    gridsizes = (int *)calloc(nprocs, sizeof(int));
    for(i=1; i < nprocs; ++i){
      MPE_Decomp1d(nx, nprocs, i, &s, &e );
      gridsizes[i] = (e - s + 1) + 2;
      
      subgrid = (double **)malloc(gridsizes[i] * sizeof(double *));
      tmp    = (double *)malloc(gridsizes[i]*(nx+2)*sizeof(double));
      for(j=0; j < gridsizes[i]; ++j){
	subgrid[j] = &tmp[j*(nx+2)];
      }

      /* for(k=0; k < gridsizes[i]; ++k){ */
      /* 	for(j=0; j < (nx+2); ++j){ */
      /* 	  printf("--%lf ",subgrid[k][j]); */
      /* 	} */
      /* 	printf("\n"); */
      /* } */

      MPI_Recv(tmp, gridsizes[i]*(nx+2), MPI_DOUBLE, i, i, comm, &status);

      for(k=s; k < e+2; ++k){
      	for(j=0; j < (nx+2); ++j){
	  g[k][j] = subgrid[k - s + 1][j];
      	}
      }

      free(subgrid);
      free(tmp);
    }
    free(gridsizes);

  } else {
    MPE_Decomp1d(nx, nprocs, myid, &s, &e );
    localcols =  (e - s + 1) + 2;
    subgrid = (double **)malloc(localcols * sizeof(double *));
    tmp    = (double *)malloc(localcols*(nx+2)*sizeof(double));
    for(i=0; i < localcols; ++i){
      subgrid[i] = &tmp[i*(nx+2)];
    }
    for(i=s-1; i<e+2; ++i){
      subgrid_index = i - s + 1;
      if( subgrid_index < 0){
	fprintf(stderr,"==============> Error: index cannot be negative: %d\n",subgrid_index);
	exit(6);
      }
      for(j=0; j < nx+2; ++j){
	subgrid[subgrid_index][j] = a[i][j];
      }
    }

    MPI_Send(tmp, localcols * (nx+2), MPI_DOUBLE, 0, myid, comm);

    free(subgrid);
    free(tmp);
  }

}

void set_full_grid_func(double u[][maxn], int nx, int ny, int s, int e,
		       double (*gf)(int xind, int yind, int nx, int ny, int s, int e))
{
  int i,j;

  for(i=0;i<maxn; i++){
    for(j=0;j<maxn;j++){
      u[i][j] = gf(i, j, nx, ny, s, e);
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
void copy_full_grid(double a[][maxn], double b[][maxn])
{
  int i, j;

  for(i=0; i < maxn; ++i){
    for(j=0; j < maxn; ++j){
      a[i][j] = b[i][j];
    }
  }
}
