/*
 * Include file for calculating orthant probabilities
 *
 * Stored in
 *   "orthant.h"
 *
 * Last modified:
 *   2003-05-06
 *
 * (c) 2001-2003 T. Miwa
 *
 */

/*
 * Modifications for R package `mvtnorm' by
 * Xuefei Mi <mi@biostat.uni-hannover.de> and 
 * Torsten Hothorn <Torsten.Hothorn@R-project.org>
 *
 */

/* include R header files */
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>

#define MAXM    20    /* maximum dimension size  */
#define MAXGRD  4097  /* maximum number of grid points */

/* normal density function */
#define nrml_dn(X)  dnorm(X, 0, 1, 0)

/* 1 - normal cumulative distribution function */
#define nrml_cd(X)  pnorm(X, 0, 1, 1, 0)

/* Grid information
 * The values are calculated in gridcalc.c
 *   and passed to "orschm.c" throuth "orthant.c".
 */
struct GRID{
  int n;                /* even number of grid points */
  double z[MAXGRD];     /* grid points */
  double w[MAXGRD];     /* width, w[j]=z[j]-z[j-1] */
  double p[MAXGRD];     /* lower normal prob at z[j] */
  double d[MAXGRD];     /* normal density at z[j] */
  double w2[MAXGRD];    /* w squared */
  double w3[MAXGRD];    /* w cubed */
  double q[MAXGRD][4];  /* integral (x - z[j-1])^i * phi(x) */
};
