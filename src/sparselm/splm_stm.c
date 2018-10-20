/////////////////////////////////////////////////////////////////////////////////
////// 
//////  ST sparse matrices manipulation routines
//////  Copyright (C) 2008  Manolis Lourakis (lourakis@ics.forth.gr)
//////  Institute of Computer Science, Foundation for Research & Technology - Hellas
//////  Heraklion, Crete, Greece.
//////
//////  This program is free software; you can redistribute it and/or modify
//////  it under the terms of the GNU General Public License as published by
//////  the Free Software Foundation; either version 2 of the License, or
//////  (at your option) any later version.
//////
//////  This program is distributed in the hope that it will be useful,
//////  but WITHOUT ANY WARRANTY; without even the implied warranty of
//////  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//////  GNU General Public License for more details.
//////
/////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "splm.h"


/* allocate a sparse ST matrix, no values */
void splm_stm_alloc(struct splm_stm *sm, MKL_INT nr, MKL_INT nc, MKL_INT maxnnz)
{
  sm->nr=nr;
  sm->nc=nc;
  sm->maxnnz=maxnnz;
  sm->nnz=0;
  sm->val=NULL;

  sm->rowidx=(MKL_INT *)malloc(maxnnz*sizeof(MKL_INT));
  sm->colidx=(MKL_INT *)malloc(maxnnz*sizeof(MKL_INT));
  if(!sm->rowidx || !sm->colidx){
    fprintf(stderr, "memory allocation request failed in splm_stm_alloc() [nr=%Id, nc=%Id, maxnnz=%Id]\n", nr, nc, maxnnz);
    exit(1);
  }
}

/* allocate a sparse ST matrix, with values */
void splm_stm_allocval(struct splm_stm *sm, MKL_INT nr, MKL_INT nc, MKL_INT maxnnz)
{
  sm->nr=nr;
  sm->nc=nc;
  sm->maxnnz=maxnnz;
  sm->nnz=0;

  sm->rowidx=(MKL_INT *)malloc(maxnnz*sizeof(MKL_INT));
  sm->colidx=(MKL_INT *)malloc(maxnnz*sizeof(MKL_INT));
  sm->val=(double *)malloc(maxnnz*sizeof(double));
  if(!sm->rowidx || !sm->colidx || !sm->val){
    fprintf(stderr, "memory allocation request failed in splm_stm_allocval() [nr=%Id, nc=%Id, maxnnz=%Id]\n", nr, nc, maxnnz);
    exit(1);
  }
}


/* free a sparse CCS matrix */
void splm_stm_free(struct splm_stm *sm)
{
  sm->nr=sm->nc=sm->nnz=sm->maxnnz=-1;
  free(sm->rowidx);
  free(sm->colidx);
  if(sm->val) free(sm->val);

  sm->rowidx=sm->colidx=NULL;
  sm->val=NULL;
}

/* add an element to a triplet matrix.
 * returns 1 if successful, 0 otherwise
 */
MKL_INT splm_stm_nonzero(struct splm_stm *sm, MKL_INT i, MKL_INT j)
{
  if(sm->nnz==sm->maxnnz) return 0; // no space available

  if(i>=sm->nr || j>=sm->nc) return 0; // out of bounds

  sm->rowidx[sm->nnz]=i;
  sm->colidx[sm->nnz++]=j;

  return 1; /* ok */
}

/* as above with a numerical value */
MKL_INT splm_stm_nonzeroval(struct splm_stm *sm, MKL_INT i, MKL_INT j, double val)
{
  if(sm->nnz==sm->maxnnz) return 0; // no space available

  if(i>=sm->nr || j>=sm->nc) return 0; // out of bounds

  //if(sm->val==NULL) return 0; // no values allocated
  
  sm->rowidx[sm->nnz]=i;
  sm->colidx[sm->nnz]=j;
  sm->val[sm->nnz++]=val;

  return 1; /* ok */
}


#if 0
static void splm_stm_print(struct splm_stm *sm, FILE *fp)
{
register MKL_INT i;

  fprintf(fp, "matrix is %dx%Id, %Id non-zeros\n", sm->nr, sm->nc, sm->nnz);
  fprintf(fp, "\nrowidx, colidx: ");
  if(sm->val)
    for(i=0; i<sm->nnz; ++i)
      fprintf(fp, "%Id %Id %g, ", sm->rowidx[i], sm->colidx[i], sm->val[i]);
  else
    for(i=0; i<sm->nnz; ++i)
      fprintf(fp, "%Id %Id, ", sm->rowidx[i], sm->colidx[i]);
  fprintf(fp, "\n");
}
#endif

/* convert a matrix from the ST format to CCS.
 * No checks/actions (e.g., summation) for duplicate indices!
 */
void splm_stm2ccsm(struct splm_stm *st, struct splm_ccsm *ccs)
{
register MKL_INT i, j, k, l;
MKL_INT nr, nc, nnz;
MKL_INT *colidx, *colptr, *ccsrowidx, *strowidx;
MKL_INT *colcounts; // counters for the number of nonzeros in each column
MKL_INT *tmpidx, imax;
register double *ccsv;

  nr=st->nr; nc=st->nc;
  nnz=st->nnz;

  /* allocate only if ccs->val is NULL */
  if(!ccs->val) splm_ccsm_alloc(ccs, nr, nc, nnz);

  ccsv=ccs->val;
  ccs->nr=nr; ccs->nc=nc; // ensure that ccs has the correct dimensions

  if((colcounts=(MKL_INT *)calloc(nc, sizeof(MKL_INT)))==NULL){ // init to zero
    fprintf(stderr, "memory allocation request failed in splm_stm2ccsm() [nr=%Id, nc=%Id, nnz=%Id]\n", nr, nc, nnz);
    exit(1);
  }

  colidx=st->colidx; strowidx=st->rowidx;
  colptr=ccs->colptr; ccsrowidx=ccs->rowidx;

  /* 1st pass: count #nonzeros in each column */
  for(i=nnz; i-->0;  )
    ++(colcounts[colidx[i]]);

  /* 2nd pass: copy every nonzero to its column into the CCS structure */
  for(j=k=0; j<nc; ++j){
    colptr[j]=k;
    k+=colcounts[j];
  }
  colptr[nc]=nnz;

  /* colcounts[j] will count the #nonzeros in col. j seen before the current row */
  memset(colcounts, 0, nc*sizeof(MKL_INT)); /* reset to zero */

  for(j=0; j<nnz; ++j){
    l=colidx[j];
    k=colptr[l];
    k+=colcounts[l];
    ++(colcounts[l]);

    /* note that ccsrowidx[k] is set below to the index
     * of the triplet in st and *not* its row number!
     */
    ccsrowidx[k]=j;
  }
  free(colcounts); colcounts=NULL;

  /* 3rd pass: sort colums into the CCS structure according to their row index */
  /* allocate work arrays */
  tmpidx=(MKL_INT *)malloc(nr*sizeof(MKL_INT));
  if(!tmpidx){
    fprintf(stderr, "memory allocation request failed in splm_stm2ccsm()\n");
    exit(1);
  }

  /* sort row indices using linear bucket sort */
  for(i=nr; i-->0; ) tmpidx[i]=-1;
  
  if(st->val){ // have values
    register double *stv=st->val;

    for(j=0; j<nc; ++j){
      imax=colptr[j+1];
      for(i=colptr[j]; i<imax; ++i){
        k=ccsrowidx[i];
        tmpidx[strowidx[k]]=k;
      }

      for(i=0, k=colptr[j]; i<nr; ++i)
        if(tmpidx[i]>=0){
          ccsv[k]=stv[tmpidx[i]];
          tmpidx[i]=-1; /* restore */
          ccsrowidx[k++]=i;
        }
    }
  }
  else{ // no values
    for(j=0; j<nc; ++j){
      imax=colptr[j+1];
      for(i=colptr[j]; i<imax; ++i){
        tmpidx[strowidx[ccsrowidx[i]]]=1;
      }

      for(i=0, k=colptr[j]; i<nr; ++i)
        if(tmpidx[i]>0){
          tmpidx[i]=-1; /* restore */
          ccsrowidx[k++]=i;
        }
    }
  }

  free(tmpidx);
}

/* interface to splm_stm2ccsm() resembling matlab's sparse(i,j,s,m,n,nzmax)
 * Nonzero element k is specified by (rowidx[k], colidx[k]) and is equal to s[k]
 *
 * See http://www.mathworks.com/help/techdoc/ref/sparse.html
 */
void splm_tri2ccsm(MKL_INT *i, MKL_INT *j, double *s, MKL_INT m, MKL_INT n, MKL_INT nzmax, struct splm_ccsm *ccs)
{
struct splm_stm st;

  st.nr=m; st.nc=n;
  st.nnz=st.maxnnz=nzmax;
  st.rowidx=i;
  st.colidx=j;
  st.val=s;

  splm_stm2ccsm(&st, ccs);
}


/* convert a matrix from the ST format to CRS.
 * No checks/actions for duplicate indices!
 */
void splm_stm2crsm(struct splm_stm *st, struct splm_crsm *crs)
{
register MKL_INT i, j, k, l;
MKL_INT nr, nc, nnz;
MKL_INT *rowidx, *rowptr, *crscolidx, *stcolidx;
MKL_INT *rowcounts; // counters for the number of nonzeros in each row
MKL_INT *tmpidx, jmax;
register double *crsv;

  nr=st->nr; nc=st->nc;
  nnz=st->nnz;

  /* allocate only if crs->val is NULL */
  if(!crs->val) splm_crsm_alloc(crs, nr, nc, nnz);

  crsv=crs->val;
  crs->nr=nr; crs->nc=nc; // ensure that crs has the correct dimensions

  if((rowcounts=(MKL_INT *)calloc(nr, sizeof(MKL_INT)))==NULL){ // init to zero
    fprintf(stderr, "memory allocation request failed in splm_stm2crsm() [nr=%Id, nc=%Id, nnz=%Id]\n", nr, nc, nnz);
    exit(1);
  }

  rowidx=st->rowidx; stcolidx=st->colidx;
  rowptr=crs->rowptr; crscolidx=crs->colidx;

  /* 1st pass: count #nonzeros in each row */
  for(i=nnz; i-->0;  )
    ++(rowcounts[rowidx[i]]);

  /* 2nd pass: copy every nonzero to its row into the CRS structure */
  for(i=k=0; i<nr; ++i){
    rowptr[i]=k;
    k+=rowcounts[i];
  }
  rowptr[nr]=nnz;

  /* rowcounts[i] will count the #nonzeros in row i seen before the current column */
  memset(rowcounts, 0, nr*sizeof(MKL_INT)); /* initialize to zero */

  for(i=0; i<nnz; ++i){
    l=rowidx[i];
    k=rowptr[l];
    k+=rowcounts[l]++;

    /* note that crscolidx[k] is set below to the index
     * of the triplet in st and *not* its column number!
     */
    crscolidx[k]=i;
  }
  free(rowcounts); rowcounts=NULL;

  /* 3rd pass: sort rows into the CRS structure according to their column index */
  /* allocate work arrays */
  tmpidx=(MKL_INT *)malloc(nc*sizeof(MKL_INT));
  if(!tmpidx){
    fprintf(stderr, "memory allocation request failed in splm_stm2crsm()\n");
    exit(1);
  }

  /* sort column indices using linear bucket sort */
  for(j=nc; j-->0;  ) tmpidx[j]=-1;

  if(st->val){ // have values
    register double *stv=st->val;

    for(i=0; i<nr; ++i){
      jmax=rowptr[i+1];
      for(j=rowptr[i]; j<jmax; ++j){
        k=crscolidx[j];
        tmpidx[stcolidx[k]]=k;
      }

      for(j=0, k=rowptr[i]; j<nc; ++j)
        if(tmpidx[j]>=0){
          crsv[k]=stv[tmpidx[j]];
          tmpidx[j]=-1; /* restore */
          crscolidx[k++]=j;
        }
    }
  }
  else{ // no values
    for(i=0; i<nr; ++i){
      jmax=rowptr[i+1];
      for(j=rowptr[i]; j<jmax; ++j){
        tmpidx[stcolidx[crscolidx[j]]]=1;
      }

      for(j=0, k=rowptr[i]; j<nc; ++j)
        if(tmpidx[j]>0){
          tmpidx[j]=-1; /* restore */
          crscolidx[k++]=j;
        }
    }
  }

  free(tmpidx);
}

/* interface to splm_stm2crsm() resembling matlab's sparse(i,j,s,m,n,nzmax)
 */
void splm_tri2crsm(MKL_INT *i, MKL_INT *j, double *s, MKL_INT m, MKL_INT n, MKL_INT nzmax, struct splm_crsm *crs)
{
struct splm_stm st;

  st.nr=m; st.nc=n;
  st.nnz=st.maxnnz=nzmax;
  st.rowidx=i;
  st.colidx=j;
  st.val=s;

  splm_stm2crsm(&st, crs);
}

