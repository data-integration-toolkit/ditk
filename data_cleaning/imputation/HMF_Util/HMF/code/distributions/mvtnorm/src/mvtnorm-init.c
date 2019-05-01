
#include "mvtnorm.h"
#include <R_ext/Rdynload.h>

void C_mvtdst(int *n, int *nu, double *lower, double *upper,
              int *infin, double *corr, double *delta,
              int *maxpts, double *abseps, double *releps,
              double *error, double *value, int *inform, int *rnd)
{

    if (rnd[0]) GetRNGstate();

    /* call FORTRAN subroutine */
    F77_CALL(mvtdst)(n, nu, lower, upper, 
                     infin, corr, delta,
                     maxpts, abseps, releps, 
                     error, value, inform);

    if (rnd[0]) PutRNGstate();

}

// TVPACK n=3
void C_tvtlr(int *NU, double *H, double *R, double *EPSI, double *TVTL) {

    F77_CALL(tvtlrcall)(NU, H, R, EPSI, TVTL); 
}

// TVPACK n=2
void C_bvtlr(int *NU, double *DH, double *DK, double *R, double *BVTL) {

    F77_CALL(bvtlrcall)(NU, DH, DK, R, BVTL );
}


static const R_CMethodDef cMethods[] = {
    {"C_mvtdst", (DL_FUNC) &C_mvtdst, 14, (R_NativePrimitiveArgType[14]){INTSXP, INTSXP, REALSXP, REALSXP, 
                                           INTSXP, REALSXP, REALSXP, 
                                           INTSXP, REALSXP, REALSXP, 
                                           REALSXP, REALSXP, INTSXP, INTSXP}}, 
    {"C_tvtlr", (DL_FUNC) &C_tvtlr, 5, (R_NativePrimitiveArgType[13]){INTSXP, REALSXP, REALSXP, REALSXP, REALSXP}},
    {"C_bvtlr", (DL_FUNC) &C_bvtlr, 5, (R_NativePrimitiveArgType[13]){INTSXP, REALSXP, REALSXP, REALSXP, REALSXP}},
    {NULL, NULL, 0}
};

static const R_CallMethodDef callMethods[] = {
    {"C_miwa", (DL_FUNC) &C_miwa, 5},
    {NULL, NULL, 0}
};


void R_init_mvtnorm(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, callMethods, cMethods, NULL);
    R_RegisterCCallable("mvtnorm", "C_mvtdst", (DL_FUNC) &C_mvtdst);
}
