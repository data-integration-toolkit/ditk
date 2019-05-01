#include <R.h>
#include <Rmath.h>
#include <R_ext/RS.h>

void F77_SUB(rndstart)(void) { GetRNGstate(); }
void F77_SUB(rndend)(void) { PutRNGstate(); }
double F77_SUB(normrnd)(void) { return norm_rand(); }
double F77_SUB(unifrnd)(void) { return unif_rand(); }

double F77_SUB(pnormr)(double *x, double *mu, double *sigma, int *lower_tail, int *log_p) { return pnorm(*x, *mu, *sigma, *lower_tail, *log_p); }
double F77_SUB(qnormr)(double *p, double *mu, double *sigma, int *lower_tail, int *log_p) { return qnorm(*p, *mu, *sigma, *lower_tail, *log_p); }
