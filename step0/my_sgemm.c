/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_sgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab sgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */
 

#include <bl_sgemm.h>
#include "bl_config.h"

void bl_sgemm(
    int    m,
    int    n,
    int    k,
    float *A,
    int    lda,
    float *B,
    int    ldb,
    float *C,
    int    ldc
)
{
  int i, j, p;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "bl_sgemm(): early return\n" );
    return;
  }

#ifdef ROW_MAJOR
  assert(lda == k);
  assert(ldb == n);
  assert(ldc == n);
  // ipj mode
  for ( i = 0; i < m; i ++ ) {              // Start 2-th loop
      for ( p = 0; p < k; p ++ ) {          // Start 1-st loop
          for ( j = 0; j < n; j ++ ) {      // Start 0-nd loop
              C[ i * ldc + j ] += A[ i * lda + p ] * B[ p * ldb + j ];
          }                                 // End   0-th loop
      }                                     // End   1-st loop
  }                                         // End   2-nd loop
#else /* COLUMN_MAJOR */
  assert(lda == m);
  assert(ldb == k);
  assert(ldc == m);
  // jpi mode
  for ( j = 0; j < n; j ++ ) {              // Start 2-nd loop
      for ( p = 0; p < k; p ++ ) {          // Start 1-st loop
          for ( i = 0; i < m; i ++ ) {      // Start 0-th loop
              C[ j * ldc + i ] += A[ p * lda + i ] * B[ j * ldb + p ];
          }                                 // End   0-th loop
      }                                     // End   1-st loop
  }                                         // End   2-st loop
#endif
}


