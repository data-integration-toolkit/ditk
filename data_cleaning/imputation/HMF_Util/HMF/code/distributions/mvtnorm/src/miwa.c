
/*
 * Test program for calculating
 *    orthant probabilities.
 *
 * Command format:
 *   orthant m [ngrd [rho [h]]]
 *
 * Required functions
 *   void gridcalc()
 *   double orthant()
 *
 * Note
 *   An exact value is given only for rho=0.5 and h=0.
 *
 */

#include  "miwa.h"
#define UEPS    1.0e-8
#define PEPS    1.0e-8

#define SMLHGRD 16  /* small number of grid points */

#define REPS    (1.0e-6)  /* rho < REPS means rho=0 */

/* coefficients b[] for cubic polynomial */
static void b_calc(int j, struct GRID *g, double *f, double *df,
                   double *b)
{
  b[0] = f[j-1];
  b[1] = df[j-1];
  b[2] = 3.0*(-f[j-1]+f[j])/g->w2[j] - (2.0*df[j-1]+df[j])/g->w[j];
  b[3] = 2.0*(f[j-1]-f[j])/g->w3[j] + (df[j-1]+df[j])/g->w2[j];
}


/* integral of f(x)*phi(x) from g->z[j-1] to g->z[j-1]+dz */
static double dlt_f(int j, struct GRID *g,
                    double np, double nd, double dz,
                    double *b)
{
  double q0, q1, q2, q3;

  q0 = np - g->p[j-1];
  q1 = -nd + g->d[j-1] - g->z[j-1]*q0;
  q2 = -dz*nd - g->z[j-1]*q1 + q0;
  q3 = -dz*dz*nd - g->z[j-1]*q2 + 2.0*q1;
  return (b[0]*q0 + b[1]*q1 + b[2]*q2 + b[3]*q3);
}


static double orschm(int m, double *r, double *h, struct GRID *g)
{
  static int    id[MAXM][MAXGRD];
  static double c[MAXM], d[MAXM], b[MAXGRD][4], fgrd[MAXGRD];
  static double z[MAXM][MAXGRD], np[MAXM][MAXGRD], nd[MAXM][MAXGRD];
  static double f[MAXGRD], df[MAXGRD];
  int    i, j, k, ngrd=g->n;
  double detr, detr1=1.0, dz, fbase;

  /* Cholesky decompositon */
  for(i=1; i<m; i++){

    /* detr=det/det1: determinant ratio */
    detr = 1.0 - r[i-1]*r[i-1]/detr1;
    c[i] = h[i]/sqrt(detr);
    d[i] = -r[i-1]/sqrt(detr1*detr);
    detr1 = detr;
  }


  /* normal densities and probabilities at upper limits
   * z[i][j]:  upper limit of integration for the next stage
   * nd[i][j]: normal density at z[i][j]
   * np[i][j]: lower probability at z[i][j]
   */
  for(i=1; i < m-1; i++)
    for(j=0; j <= ngrd; j++){
      z[i][j] = c[i] + d[i]*g->z[j];
      nd[i][j] = nrml_dn(z[i][j]);
      np[i][j] = nrml_cd(z[i][j]);
    }

  /* Check where z[i][k]=c[i]+d[i]*(g->z[k]) is located.
   *   id[i][k] = j      if g->z[j-1] < z[i][k] <= g->z[j]
   *   id[i][k] = 0      if z[i][k] <= g->z[0]=-8
   *   id[i][k] = ngrd+1 if 8=g->z[ngrd] < z[i][k]
   */
  for(i=1; i < m-1; i++){
    if(d[i] > 0){
      for(j=0, k=0; j <= ngrd; j++)
        for( ; z[i][k] <= g->z[j] && k <= ngrd; k++)
          id[i][k] = j;
      for( ; k <= ngrd; k++)
        id[i][k] = ngrd+1;
    }
    else{
      for(j=0, k=ngrd; j <= ngrd; j++)
/* Tetsuhisa Thu, 16 Jan 2014; was  for( ; z[i][k] <= g->z[j] && k >= 0; k--) */
        for( ; k >= 0 && z[i][k] <= g->z[j]; k--)
          id[i][k] = j;
      for( ; k >= 0; k--)
        id[i][k] = ngrd+1;
    }
  }

  /* first stage: i=m-1 */
  for(j=0; j <= ngrd; j++){
    z[m-1][j] = c[m-1] + d[m-1]*g->z[j];
    f[j] = nrml_cd(z[m-1][j]);
    df[j] = d[m-1] * nrml_dn(z[m-1][j]);
  }

  /* intermediate stages: i=m-2, ..., 1 */
  for(i=m-2; i > 0; i--){

    /* integrated values fgrd[j] at g->z[j] */
    for(j=1, fgrd[0]=0.0; j <= ngrd; j++){
      b_calc(j, g, f, df, b[j]);
      fgrd[j] = fgrd[j-1]
        + b[j][0] * g->q[j][0] + b[j][1] * g->q[j][1]
        + b[j][2] * g->q[j][2] + b[j][3] * g->q[j][3];
    }

    for(k=0; k <= ngrd; k++){
      /* lower than g->z[0]=-8 */
      if(id[i][k] < 1)
        f[k] = df[k] = 0.0;
      /* greater than g->z[ngrd]=8 */
      else if(id[i][k] > ngrd){
        df[k] = 0.0;
        f[k] = fgrd[ngrd];
      }
      /* between g->z[0] and g->z[ngrd] */
      else{
        j = id[i][k];
        dz = z[i][k] - g->z[j-1];
        df[k] =  nd[i][k] * d[i]
          * (b[j][0] + dz*(b[j][1] + dz*(b[j][2] + dz*b[j][3])));
        f[k] = fgrd[j-1]
          + dlt_f(j, g, np[i][k], nd[i][k], dz, b[j]);
      }
    }
  }

  /* last stage: h[0] = c[0] */
  for(j=1, fbase=0.0; j <= ngrd && g->z[j] <= h[0]; j++){
    b_calc(j, g, f, df, b[j]);
    fbase += b[j][0] * g->q[j][0] + b[j][1] * g->q[j][1]
      + b[j][2] * g->q[j][2] + b[j][3] * g->q[j][3];
  }
  if(j <= ngrd && g->z[j-1] < h[0]){
    b_calc(j, g, f, df, b[j]);
    np[0][0] = nrml_cd(h[0]);
    nd[0][0] = nrml_dn(h[0]);
    dz = h[0] - g->z[j-1];
    fbase += dlt_f(j, g, np[0][0], nd[0][0], dz, b[j]);
  }

  return (fbase);
}


static double orthant(int m, double r[][MAXM][MAXM], double h[][MAXM],
		      int *ncone, struct GRID *grid)
{
  int     i, j, u, v, ns, nzs, stg, srch, plus;
  int     nz[MAXM][MAXM], sgn[MAXM][MAXM], nxt[MAXM], dlt[MAXM];
  double  rvec[MAXM], hvec[MAXM], c[MAXM];
  double  p=0.0, r1k, r1ik, rik, ruk;

  /* initialisation */
  stg = 0;      /* stage pointer: 0 <= stg <= m-2 */
  srch = 1;     /* swich for searching non-zero coefficients */
  dlt[0] = 1;   /* plus or minus contribution of each cone */
  *ncone = 0;   /* number of sub-cones */

  /* rvec[]: sub-diagonal cor coef for orthoscheme prob
   * hvec[]: upper bound vecter for orthoscheme prob
   */
  hvec[0] = h[0][0];

  while(stg >= 0){

    /* calculate orthoscheme probability */
    if(stg == m-2){
      rvec[stg] = r[stg][stg][stg+1];
      hvec[stg+1] = h[stg][stg+1];
      p += dlt[stg]*orschm(m, rvec, hvec, grid);
      (*ncone)++;
      srch = 0;
      stg--;
    }

    /* search for non-zero cor coeff rho[] */
    else if(srch == 1){
      for(plus=nz[stg][0]=0, j=1, i=stg+1; i < m; i++){
        if(r[stg][stg][i] > REPS){
          plus = 1;         /* plus=0 if no positive rho's */
          nz[stg][0]++;     /* nz[stg][0] = no of non-zero rho's */
          nz[stg][j] = i;   /* address of non-zero rho */
          sgn[stg][j] = 1;  /* sign of rho */
          j++;
        }
        else if(r[stg][stg][i] < -REPS){
          nz[stg][0]++;
          nz[stg][j] = i;
          sgn[stg][j] = -1;
          j++;
        }
      }

      if(nz[stg][0] == 0)
        nxt[stg] = 0;
      else{
        nxt[stg] = 1;
        /* if all the non-rero rho's are negative */
        if(plus == 0)
          for(j=1; j <= nz[stg][0]; j++)
            sgn[stg][j] = 1;
      }

      srch = 0;
    }

    /* back to the previous stage */
    else if(nxt[stg] > nz[stg][0])
      stg--;

    /* if all cor coeff's are zero */
    else if(nz[stg][0] == 0){
      rvec[stg] = 0.0;
      hvec[stg+1] = h[stg][stg+1];
      for(i=stg+2; i < m; i++)
        h[stg+1][i] = h[stg][i];
      for(i=stg+1; i < m-1; i++)
        for(j=i+1; j < m; j++)
          r[stg+1][i][j] = r[stg][i][j];
      dlt[stg+1] = dlt[stg];
      nxt[stg]++;
      stg++;
      srch = 1;
    }

    /* calculate cor coeff's for the next stage */
    else{
      ns=nxt[stg];
      nzs=nz[stg][ns];

      r1k = r[stg][stg][nzs];
      rvec[stg] = sgn[stg][ns] * r1k;
      hvec[stg+1] = sgn[stg][ns] * h[stg][nzs];
      for(i=stg+1, j=stg+2; j < m; i++, j++){
        if(i == nzs)
          i++;
        r1ik = r[stg][stg][i]/r1k;
        if (i < nzs)
          rik = r[stg][i][nzs];
        else
          rik = r[stg][nzs][i];

        c[j] = sqrt(1.0 - 2.0*r1ik*rik + r1ik*r1ik);
        h[stg+1][j] = (h[stg][i] - r1ik*h[stg][nzs])/c[j];
        r[stg+1][stg+1][j] = sgn[stg][ns]/c[j]*(rik - r1ik);
      }

      for(i=stg+1, j=stg+2; j < m-1; i++, j++){
        if(i == nzs)
          i++;
        for(u=i+1, v=j+1; v < m; u++, v++){
          if(u == nzs)
            u++;
          if (i < nzs)
            rik = r[stg][i][nzs];
          else
            rik = r[stg][nzs][i];
          if (u < nzs)
            ruk = r[stg][u][nzs];
          else
            ruk = r[stg][nzs][u];

          r[stg+1][j][v] =
            (r[stg][i][u]
             - r[stg][stg][u]/r1k*rik - r[stg][stg][i]/r1k*ruk
             + r[stg][stg][i]*r[stg][stg][u]/r1k/r1k) /c[j]/c[v];
        }
      }

      dlt[stg+1] = sgn[stg][ns]*dlt[stg];
      nxt[stg]++;
      stg++;
      srch = 1;
    }
  }

  return (p);
}


#define MAXITR  51  /* maximum number of iterations */


static double nrml_lq(double p, double ueps, double peps, int *itr)
{
  double y, u, f, f1, f2, r, delta;

  y = -log(4.0 * p * (1.0 - p));
  u = sqrt(y * (2.0611786 - 5.7262204/(y+11.640595)));
  if(p < 0.5)
    u = -u;

  for(*itr=1, delta=0.0; *itr < MAXITR; (*itr)++){
    f = nrml_cd(u) - p;
    if(fabs(delta) < ueps && fabs(f) < peps)
      break;

    f1 = exp(-0.5*u*u) * 0.39894228040143268;
    f2 = -u * f1;
    r = f1*f1 - 2*f*f2;
    if(r > 0)
      delta = 2*f / (-f1-sqrt(r));
    else
      delta = -f1/f2;

    u += delta;
  }
  return(u);
}


static void gridcalc(struct GRID *g)
{
  int     hgrd=(g->n)/2, ngrd=2*hgrd, nres=(hgrd<100)?3:6;
  int     i, itr;
  double  pdelta;

  g->z[0] = -8.0;
  g->z[hgrd] = 0.0;
  g->z[ngrd] = 8.0;
  g->p[0] = 0.0;
  g->p[hgrd] = 0.5;
  g->p[ngrd] = 1.0;
  g->d[0] = 0.0;
  g->d[hgrd] = 0.3989422804014327;
  g->d[ngrd] = 0.0;


  /* If #{grid points} is very small,
   *   integrate between [-5, 5].
   */
  if(hgrd < SMLHGRD){
    g->z[0] = -5.0;
    g->z[ngrd] = 5.0;
    nres = 0;
  }

  pdelta = (nrml_cd(2.5)-0.5) / (hgrd-nres);

  for(i=1; i < hgrd-nres; i++){
    g->z[hgrd+i] = 2.0*nrml_lq(0.5+i*pdelta, UEPS, PEPS, &itr);
    g->z[hgrd-i] = - g->z[hgrd+i];
    g->p[hgrd+i] = nrml_cd(g->z[hgrd+i]);
    g->p[hgrd-i] = 1.0 - g->p[hgrd+i];
    g->d[hgrd-i] = g->d[hgrd+i] = nrml_dn(g->z[hgrd+i]);
  }

  for(i=0; i < nres; i++){
    g->z[ngrd-nres+i] = 5.0 + i*3.0/nres;
    g->z[nres-i] = - g->z[ngrd-nres+i];
    g->p[ngrd-nres+i] = nrml_cd(g->z[ngrd-nres+i]);
    g->p[nres-i] = 1.0 - g->p[ngrd-nres+i];
    g->d[nres-i] = g->d[ngrd-nres+i] = nrml_dn(g->z[ngrd-nres+i]);
  }

  g->w[0] = g->w2[0] = g->w3[0] = 0.0;
  g->q[0][0] = g->q[0][1] = g->q[0][2] = g->q[0][3] = 0.0;
  for(i=1; i <= ngrd; i++){
    g->w[i] = g->z[i] - g->z[i-1];
    g->w2[i] = g->w[i] * g->w[i];
    g->w3[i] = g->w[i] * g->w2[i];
    g->q[i][0] = g->p[i] - g->p[i-1];
    g->q[i][1] = - g->d[i] + g->d[i-1] - g->z[i-1] * g->q[i][0];
    g->q[i][2] = - g->w[i] * g->d[i]
      - g->z[i-1] * g->q[i][1] + g->q[i][0];
    g->q[i][3] = - g->w2[i] * g->d[i]
      - g->z[i-1] * g->q[i][2] + 2.0 * g->q[i][1];
    if (i == 1) Rprintf("");
  }

  g->n = ngrd;
  return;
}

static int checkall(int *vector,int length,int value)
 {
       int returnvalue=1, i = 0;
       for (i=0; i< length ; i++)
       {
           if (vector[i]!=value)
           {
             returnvalue=0;
           }
       }
      return(returnvalue);
 }


/*
 *  interface to R
 *
 *  Author: Xuefei Mi <mi@biostat.uni-hannover.de>
 *
 */

SEXP C_miwa(SEXP steps, SEXP corr, SEXP upper, SEXP lower, SEXP infin) {

    SEXP answer;
    int dim;
    /* int diml; ### was not used */
    int i,ii, j,k,l,i5,i6,i7,i8, ncone;
    int infinlength;

/*
infinvalue is used to take the value of infin.
*/


    double *dupper,*dlower, *dcorr, output,*f, r[MAXM][MAXM][MAXM], hv[MAXM][MAXM], d[MAXM][MAXM];
    int *infinvalue;
    struct GRID   grid;

    dim = LENGTH(upper);
    dupper = REAL(upper);
    dcorr = REAL(corr);
    /* diml = LENGTH(lower); ### was not used */
    dlower = REAL(lower);
    infinvalue = INTEGER(infin);
    infinlength = LENGTH(infin);


  for (i = 0; i < dim - 1; i++) {
        for(j = i + 1; j < dim; j++) {
            r[0][i][j] = dcorr[i * dim + j];

          /* debug checking for correlation matrix
            Rprintf("r %f\n", r[0][i][j]);
          */
        }
    }

    grid.n = INTEGER(steps)[0];
    gridcalc(&grid);


 PROTECT(answer = allocVector(REALSXP, 1));



/*
 branch happens here. if only one sided, then just call orthant function once

*/
 if (checkall(infinvalue,infinlength,-1)==1 )
 {
  REAL(answer)[0]=1;
 }else if (checkall(infinvalue,infinlength,0)==1 )
 {
    for (i = 0; i < dim; i++)
    {
        hv[0][i] = dupper[i];
    }
    REAL(answer)[0] = orthant(dim, r, hv, &ncone, &grid);


 }else if (checkall(infinvalue,infinlength,1)==1 )
 {
    for (i = 0; i < dim; i++)
    {
        hv[0][i] = -dlower[i];
    }
    REAL(answer)[0] = orthant(dim, r, hv, &ncone, &grid);


 } else
 {
    for (i = 0; i < dim; i++)
    {
        hv[0][i] = dupper[i];
    }
    f=dlower;

  /*
                   # circle number 0
  */
                   output= orthant(dim, r, hv, &ncone, &grid);
  /*
                   # circle number 1
  */
                   for (i = 0; i < dim; i++)
                   {
                     for (ii = 0; ii < dim; ii++)
                          {
                             d[0][ii] = dupper[ii];
                           }
                     d[0][i]=f[i];
                    output=output-orthant(dim, r, d, &ncone, &grid);
                   }
  /*
                   # circle number 2
  */
                   for (i = 0; i < (dim-1) ; i++)
                   {
                     for (j=(i+1); j < dim; j++ )
                     {
                         for (ii = 0; ii < dim; ii++)
                               {
                                  d[0][ii] = dupper[ii];
                               }
                    d[0][i]=f[i];
                    d[0][j]=f[j];
                    output=output + orthant(dim, r, d, &ncone, &grid);
                    }
                   }
  /*
                   # circle number 3
  */
                     if (dim>2)
                     {
                      for (i = 0; i < (dim-2) ; i++ )
                      {
                       for (j=(i+1); j < (dim-1); j++ )
                       {
                         for (k=(j+1); k < dim; k++)
                           {

                           for (ii = 0; ii < dim; ii++)
                               {
                                  d[0][ii] = dupper[ii];
                               }
                              d[0][i]=f[i];
                              d[0][j]=f[j];
                              d[0][k]=f[k];
                             output=output - orthant(dim, r, d, &ncone, &grid);
                           }
                        }
                       }
                     }

                     if (dim>3)
                     {
                      for (i = 0; i < (dim-3) ; i++)
                      {
                        for (j=(i+1); j < (dim-2); j++ )
                       {
                         for (k=(j+1); k < (dim-1); k++)
                           {
                            for (l=(k+1); l < (dim); l++)
                             {


                                 for (ii = 0; ii < dim; ii++)
                                     {
                                        d[0][ii] = dupper[ii];
                                     }
                                    d[0][i]=f[i];
                                    d[0][j]=f[j];
                                    d[0][k]=f[k];
                                    d[0][l]=f[l];

                                  output=output+orthant(dim, r, d, &ncone, &grid);
                              }
                           }
                        }
                       }
                     }

                    if (dim>4)
                     {

                         for (i5 = 0; i5 < (dim-4) ; i5++)
                      {
                          for (i =(i5+1); i < (dim-3) ; i++)
                      {
                        for (j=(i+1); j < (dim-2); j++ )
                       {
                         for (k=(j+1); k < (dim-1); k++)
                           {
                            for (l=(k+1); l < (dim); l++)
                             {


                                 for (ii = 0; ii < dim; ii++)
                                     {
                                        d[0][ii] = dupper[ii];
                                     }
                                    d[0][i5]=f[i5];
                                    d[0][i]=f[i];
                                    d[0][j]=f[j];
                                    d[0][k]=f[k];
                                    d[0][l]=f[l];

                            output=output-orthant(dim, r, d, &ncone, &grid);
                             }
                           }
                        }
                       }
                       }
                     }


                    if (dim>5)
                     {

                          for (i6 = 0; i6 < (dim-5) ; i6++)
                      {
                          for (i5 = (i6+1); i5 < (dim-4) ; i5++)
                      {
                          for (i =(i5+1); i < (dim-3) ; i++)
                      {
                          for (j=(i+1); j < (dim-2); j++ )
                       {
                          for (k=(j+1); k < (dim-1); k++)
                           {
                            for (l=(k+1); l < (dim); l++)
                             {

                             for (ii = 0; ii < dim; ii++)
                                     {
                                        d[0][ii] = dupper[ii];
                                     }
                                    d[0][i6]=f[i6];
                                    d[0][i5]=f[i5];
                                    d[0][i]=f[i];
                                    d[0][j]=f[j];
                                    d[0][k]=f[k];
                                    d[0][l]=f[l];

                            output=output+orthant(dim, r, d, &ncone, &grid);


                             }
                           }
                        }
                       }
                       }
                     }
                     }


                    if (dim>6)
                     {

                          for (i7 = 0; i7 < (dim-6) ; i7++)
                      {
                          for (i6 = (i7+1); i6 < (dim-5) ; i6++)
                      {
                          for (i5 = (i6+1); i5 < (dim-4) ; i5++)
                      {
                          for (i =(i5+1); i < (dim-3) ; i++)
                      {
                          for (j=(i+1); j < (dim-2); j++ )
                       {
                          for (k=(j+1); k < (dim-1); k++)
                           {
                            for (l=(k+1); l < (dim); l++)
                             {


                             for (ii = 0; ii < dim; ii++)
                                     {
                                        d[0][ii] = dupper[ii];
                                     }
                                    d[0][i7]=f[i7];
                                    d[0][i6]=f[i6];
                                    d[0][i5]=f[i5];
                                    d[0][i]=f[i];
                                    d[0][j]=f[j];
                                    d[0][k]=f[k];
                                    d[0][l]=f[l];

                            output=output-orthant(dim, r, d, &ncone, &grid);

                             }
                           }
                        }
                       }
                       }
                     }
                     }
                     }

                     if (dim>7)
                     {

                          for (i8 = 0; i8 < (dim-7) ; i8++)
                      {
                          for (i7 = (i8+1); i7 < (dim-6) ; i7++)
                      {
                          for (i6 = (i7+1); i6 < (dim-5) ; i6++)
                      {
                          for (i5 = (i6+1); i5 < (dim-4) ; i5++)
                      {
                          for (i =(i5+1); i < (dim-3) ; i++)
                      {
                          for (j=(i+1); j < (dim-2); j++ )
                       {
                          for (k=(j+1); k < (dim-1); k++)
                           {
                            for (l=(k+1); l < (dim); l++)
                             {

                             for (ii = 0; ii < dim; ii++)
                                     {
                                        d[0][ii] = dupper[ii];
                                     }
                                    d[0][i8]=f[i8];
                                    d[0][i7]=f[i7];
                                    d[0][i6]=f[i6];
                                    d[0][i5]=f[i5];
                                    d[0][i]=f[i];
                                    d[0][j]=f[j];
                                    d[0][k]=f[k];
                                    d[0][l]=f[l];

                            output=output+orthant(dim, r, d, &ncone, &grid);

                             }
                           }
                        }
                       }
                       }
                     }
                     }
                     }
                     }









     REAL(answer)[0]=output;
 }
    UNPROTECT(1);


    return(answer);
}
