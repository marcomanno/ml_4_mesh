/*
/////////////////////////////////////////////////////////////////////////////////////////////
//// 
////  Prototypes and definitions for the sparse Levenberg - Marquardt minimization algorithm
////  Copyright (C) 2005-2011  Manolis Lourakis (lourakis at ics.forth.gr)
////  Institute of Computer Science, Foundation for Research & Technology - Hellas
////  Heraklion, Crete, Greece.
////
////  This program is free software; you can redistribute it and/or modify
////  it under the terms of the GNU General Public License as published by
////  the Free Software Foundation; either version 2 of the License, or
////  (at your option) any later version.
////
////  This program is distributed in the hope that it will be useful,
////  but WITHOUT ANY WARRANTY; without even the implied warranty of
////  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
////  GNU General Public License for more details.
////
///////////////////////////////////////////////////////////////////////////////////////////////
*/

#ifndef _SPLM_H_
#define _SPLM_H_

#include "mkl.h"
#ifdef __cplusplus
extern "C" {
#endif

/* sparse direct solvers */
#define SPLM_CHOLMOD        1
#define SPLM_CSPARSE        2
#define SPLM_LDL            3
#define SPLM_UMFPACK        4
#define SPLM_MA77           5
#define SPLM_MA57           6
#define SPLM_MA47           7
#define SPLM_MA27           8
#define SPLM_PARDISO        9
#define SPLM_DSS            10
#define SPLM_SuperLU        11
#define SPLM_TAUCS          12
#define SPLM_SPOOLES        13
#define SPLM_MUMPS          14
/*#define SPLM_sparseQR     -1 */

#define SPLM_DIFF_DELTA     1E-06 /* finite differentiation minimum delta */
#define SPLM_DELTA_SCALE    1E-04 /* finite differentiation delta scale   */

#define SPLM_OPTS_SZ        6 /* max(5, 6) */
#define SPLM_INFO_SZ        10
#define SPLM_ERROR          -1
#define SPLM_INIT_MU        1E-03
#define SPLM_STOP_THRESH    1E-12
#define SPLM_VERSION        "1.3 (December 2011)"

#define SPLM_ANJAC          0 /* analytic Jacobian */
#define SPLM_ZPJAC          1 /* approximate Jacobian, only zero pattern supplied */ 
#define SPLM_NOJAC          2 /* approximate Jacobian, zero pattern to be guessed, use cautiously */ 


/* Sparse matrix representation using Compressed Row Storage (CRS) format.
 * See http://www.netlib.org/linalg/html_templates/node91.html
 */

struct splm_crsm{
    MKL_INT nr, nc;   /* #rows, #cols for the sparse matrix */
    MKL_INT nnz;      /* number of nonzero array elements */
    double *val;  /* storage for nonzero array elements. size: nnz */
    MKL_INT *colidx;  /* column indexes of nonzero elements. size: nnz */
    MKL_INT *rowptr;  /* locations in val that start a row. size: nr+1.
                   * By convention, rowptr[nr]=nnz
                   */
};

/* Sparse matrix representation using Compressed Column Storage (CCS) format.
 * See http://www.netlib.org/linalg/html_templates/node92.html
 */

struct splm_ccsm{
    MKL_INT nr, nc;   /* #rows, #cols for the sparse matrix */
    MKL_INT nnz;      /* number of nonzero array elements */
    double *val;  /* storage for nonzero array elements. size: nnz */
    MKL_INT *rowidx;  /* row indexes of nonzero elements. size: nnz */
    MKL_INT *colptr;  /* locations in val that start a column. size: nc+1.
                   * By convention, colptr[nc]=nnz
                   */
};


/* sparse LM for functions with CRS and CCS Jacobians */

extern MKL_INT sparselm_dercrs(
       void (*func)(double *p, double *hx, MKL_INT nvars, MKL_INT nobs, void *adata),
       void (*fjac)(double *p, struct splm_crsm *jac, MKL_INT nvars, MKL_INT nobs, void *adata),
       double *p, double *x, MKL_INT nvars, const MKL_INT nconvars, MKL_INT nobs, MKL_INT Jnnz, MKL_INT JtJnnz,
       MKL_INT itmax, double opts[SPLM_OPTS_SZ], double info[SPLM_INFO_SZ], void *adata);

extern MKL_INT sparselm_derccs(
       void (*func)(double *p, double *hx, MKL_INT nvars, MKL_INT nobs, void *adata),
       void (*fjac)(double *p, struct splm_ccsm *jac, MKL_INT nvars, MKL_INT nobs, void *adata),
       double *p, double *x, MKL_INT nvars, const MKL_INT nconvars, MKL_INT nobs, MKL_INT Jnnz, MKL_INT JtJnnz,
       MKL_INT itmax, double opts[SPLM_OPTS_SZ], double info[SPLM_INFO_SZ], void *adata);

extern MKL_INT sparselm_difcrs(
       void (*func)(double *p, double *hx, MKL_INT nvars, MKL_INT nobs, void *adata),
       void (*fjac)(double *p, struct splm_crsm *jac, MKL_INT nvars, MKL_INT nobs, void *adata),
       double *p, double *x, MKL_INT nvars, const MKL_INT nconvars, MKL_INT nobs, MKL_INT Jnnz, MKL_INT JtJnnz,
       MKL_INT itmax, double opts[SPLM_OPTS_SZ], double info[SPLM_INFO_SZ], void *adata);

extern MKL_INT sparselm_difccs(
       void (*func)(double *p, double *hx, MKL_INT nvars, MKL_INT nobs, void *adata),
       void (*fjac)(double *p, struct splm_ccsm *jac, MKL_INT nvars, MKL_INT nobs, void *adata),
       double *p, double *x, MKL_INT nvars, const MKL_INT nconvars, MKL_INT nobs, MKL_INT Jnnz, MKL_INT JtJnnz,
       MKL_INT itmax, double opts[SPLM_OPTS_SZ], double info[SPLM_INFO_SZ], void *adata);

/* error checking for CRS and CCS Jacobians */
extern void sparselm_chkjaccrs(
        void (*func)(double *p, double *hx, MKL_INT m, MKL_INT n, void *adata),
        void (*jacf)(double *p, struct splm_crsm *jac, MKL_INT m, MKL_INT n, void *adata),
        double *p, MKL_INT m, MKL_INT n, MKL_INT jnnz, void *adata, double *err);

extern void sparselm_chkjacccs(
        void (*func)(double *p, double *hx, MKL_INT m, MKL_INT n, void *adata),
        void (*jacf)(double *p, struct splm_ccsm *jac, MKL_INT m, MKL_INT n, void *adata),
        double *p, MKL_INT m, MKL_INT n, MKL_INT jnnz, void *adata, double *err);

/* CRS sparse matrices manipulation routines */
extern void splm_crsm_alloc(struct splm_crsm *sm, MKL_INT nr, MKL_INT nc, MKL_INT nnz);
extern void splm_crsm_alloc_novalues(struct splm_crsm *sm, MKL_INT nr, MKL_INT nc, MKL_INT nnz);
extern void splm_crsm_alloc_values(struct splm_crsm *sm);
extern void splm_crsm_realloc_novalues(struct splm_crsm *sm, MKL_INT nr, MKL_INT nc, MKL_INT nnz);
extern void splm_crsm_free(struct splm_crsm *sm);
extern MKL_INT splm_crsm_elmidx(struct splm_crsm *sm, MKL_INT i, MKL_INT j);
extern MKL_INT splm_crsm_elmrow(struct splm_crsm *sm, MKL_INT idx);
extern MKL_INT splm_crsm_row_elmidxs(struct splm_crsm *sm, MKL_INT i, MKL_INT *vidxs, MKL_INT *jidxs);
extern MKL_INT splm_crsm_row_maxnelms(struct splm_crsm *sm);
extern MKL_INT splm_crsm_col_elmidxs(struct splm_crsm *sm, MKL_INT j, MKL_INT *vidxs, MKL_INT *iidxs);
extern void splm_crsm2ccsm(struct splm_crsm *crs, struct splm_ccsm *ccs);
extern void splm_crsm_row_sort(struct splm_crsm *sm);

/* CCS sparse matrices manipulation routines */
extern void splm_ccsm_alloc(struct splm_ccsm *sm, MKL_INT nr, MKL_INT nc, MKL_INT nnz);
extern void splm_ccsm_alloc_novalues(struct splm_ccsm *sm, MKL_INT nr, MKL_INT nc, MKL_INT nnz);
extern void splm_ccsm_alloc_values(struct splm_ccsm *sm);
extern void splm_ccsm_realloc_novalues(struct splm_ccsm *sm, MKL_INT nr, MKL_INT nc, MKL_INT nnz);
extern void splm_ccsm_free(struct splm_ccsm *sm);
extern MKL_INT splm_ccsm_elmidx(struct splm_ccsm *sm, MKL_INT i, MKL_INT j);
extern MKL_INT splm_crsm_elmcol(struct splm_ccsm *sm, MKL_INT idx);
extern MKL_INT splm_ccsm_row_elmidxs(struct splm_ccsm *sm, MKL_INT i, MKL_INT *vidxs, MKL_INT *jidxs);
extern MKL_INT splm_ccsm_col_elmidxs(struct splm_ccsm *sm, MKL_INT j, MKL_INT *vidxs, MKL_INT *iidxs);
extern MKL_INT splm_ccsm_col_maxnelms(struct splm_ccsm *sm);
extern void splm_ccsm2crsm(struct splm_ccsm *ccs, struct splm_crsm *crs);
extern MKL_INT splm_ccsm_drop_cols(struct splm_ccsm *A, MKL_INT ncols);
extern void splm_ccsm_restore_cols(struct splm_ccsm *A, MKL_INT ncols, MKL_INT ncnnz);
extern void splm_ccsm_col_sort(struct splm_ccsm *sm);

extern double splm_gettime(void);


/* Sparse matrix representation using Sparse Triplet (ST) format.
 * Note that the matrix might have an allocated size (maxnnz) larger
 * than the number of elements it actually holds (nnz). 
 * Primarily intended to be used for setting up the structure
 * of a corresponding CCS matrix.
 * See http://people.sc.fsu.edu/~jburkardt/data/st/st.html
 */

struct splm_stm{
  MKL_INT nr, nc;   /* #rows, #cols for the sparse matrix */
  MKL_INT nnz;      /* number of nonzero array elements */
  MKL_INT maxnnz;   /* maximum number of nonzero array elements the array can hold */
  MKL_INT *rowidx;  /* row indexes of nonzero elements. size: nnz */
  MKL_INT *colidx;  /* column indexes of nonzero elements. size: nnz */
  double *val;  /* nonzero elements, NULL when only structure is stored */
};

extern void splm_stm_alloc(struct splm_stm *sm, MKL_INT nr, MKL_INT nc, MKL_INT maxnnz);
extern void splm_stm_allocval(struct splm_stm *sm, MKL_INT nr, MKL_INT nc, MKL_INT maxnnz);
extern void splm_stm_free(struct splm_stm *sm);
extern MKL_INT splm_stm_nonzero(struct splm_stm *sm, MKL_INT i, MKL_INT j);
extern MKL_INT splm_stm_nonzeroval(struct splm_stm *sm, MKL_INT i, MKL_INT j, double val);
extern void splm_stm2ccsm(struct splm_stm *st, struct splm_ccsm *ccs);
extern void splm_tri2ccsm(MKL_INT *i, MKL_INT *j, double *s, MKL_INT m, MKL_INT n, MKL_INT nzmax, struct splm_ccsm *ccs);
extern void splm_stm2crsm(struct splm_stm *st, struct splm_crsm *crs);
extern void splm_tri2crsm(MKL_INT *i, MKL_INT *j, double *s, MKL_INT m, MKL_INT n, MKL_INT nzmax, struct splm_crsm *crs);


#ifdef __cplusplus
}
#endif

#endif /* _SPLM_H_ */
