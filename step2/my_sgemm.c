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

#include "bl_sgemm.h"
#include "bl_config.h"

#ifdef ROW_MAJOR
void packA_mcxkc(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        float *packA
        )
{
    int i, p;

    for (i = 0; i < m; i ++) {
      for (p = 0; p < k; p ++) {
        *packA++ = *(XA + i * ldXA + p);
      }
    }
}

void packB_kcxnc(
        int    n,
        int    k,
        float *XB,
        int    ldXB,
        float *packB
        )
{
    int j, p;

    for (p = 0; p < k; p ++) {
      for (j = 0; j < n; j ++) {
          *packB++ = *(XB + p * ldXB + j);
      }
    }
}

void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        float *packA,
        float *packB,
        float *C,
        int    ldc
        )
{
    int i, p, j;

    for (i = 0; i < m; i++) {
        for (p = 0; p < k; p++) {
            for (j = 0; j < n; j++) {
                C[i * ldc + j] += packA[i * k + p] * packB[p * n + j];
            }
        }  
    }                           
}

void bl_sgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,
        int    ldc
        )
{
    int i, j, p;
    int ic, ib, jc, jb, pc, pb;
    int ir, jr;
    float *packA, *packB;
    char *str;

    // Early return if possible
    if (m == 0 || n == 0 || k == 0) {
        printf("bl_sgemm(): early return\n");
        return;
    }

    // Allocate packing buffers
    packA = malloc(DGEMM_KC * DGEMM_MC * sizeof(float));
    packB = malloc(DGEMM_KC * DGEMM_NC * sizeof(float));

    for (ic = 0; ic < m; ic += ib) {                               // 3-rd loop around micro-kernel
        ib = min(m - ic, DGEMM_MC);
        for (pc = 0; pc < k; pc += pb) {                           // 4-th loop around micro-kernel
            pb = min(k - pc, DGEMM_KC);

            packA_mcxkc(
                        ib,
                        pb,
                        &XA[ic * lda + pc],
                        lda,
                        packA
                        );

            for (jc = 0; jc < n; jc += jb) {                // 5-th loop around micro-kernel
                jb = min(n - jc, DGEMM_NC);
                packB_kcxnc(
                    jb,
                    pb,
                    &XB[pc * ldb +  jc],
                    ldb,
                    packB
                    );

                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA,
                        packB,
                        &C[ic * ldc + jc], 
                        ldc
                    );
            }                                                      // End 3.rd loop around micro-kernel
        }                                                          // End 4.th loop around micro-kernel
    }                                                              // End 5.th loop around micro-kernel
    free( packA );
    free( packB );
}

#else /* COLUMN_MAJOR */

inline void packA_mcxkc(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        float *packA
        )
{
    int i, p;

    for (p = 0; p < k; p ++) {
        for (i = 0; i < m; i ++) {
            *packA++ = *(XA + p * ldXA + i);
        }
    }
}

inline void packB_kcxnc(
        int    n,
        int    k,
        float *XB,
        int    ldXB,
        float *packB
        )
{
    int j, p;

    for (j = 0; j < n; j ++) {
        for (p = 0; p < k; p ++) {
            *packB ++ = *(XB + j * ldXB + p);
        }
    }
}

void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        float *packA,
        float *packB,
        float *C,
        int    ldc
        )
{
    int i, p, j;

    for (j = 0; j < n; j++) {            // Start 2-nd loop
      for (p = 0; p < k; p++) {          // Start 1-st loop
        for (i = 0; i < m; i ++) {
           C[ j * ldc + i] += packA[p * m + i] * packB[j * k + p];
        }
      }                                  // End   1-st loop
  }                                      // 2-th loop around micro-kernel                                 
}

void bl_sgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,
        int    ldc
        )
{
    int i, j, p;
    int ic, ib, jc, jb, pc, pb;
    int ir, jr;
    float *packA, *packB;
    char *str;

    // Early return if possible
    if (m == 0 || n == 0 || k == 0) {
        printf("bl_sgemm(): early return\n");
        return;
    }

    // Allocate packing buffers
    packA = malloc(DGEMM_KC * DGEMM_MC * sizeof(float));
    packB = malloc(DGEMM_KC * DGEMM_NC * sizeof(float));

    for (jc = 0; jc < n; jc += jb) {                                // 5-th loop around micro-kernel
        jb = min(n - jc, DGEMM_NC);
        for (pc = 0; pc < k; pc += pb) {                          // 4-th loop around micro-kernel
            pb = min(k - pc, DGEMM_KC);

            packB_kcxnc(
                    jb,
                    pb,
                    &XB[jc * ldb +  pc],
                    ldb,
                    packB
                    );

            for ( ic = 0; ic < m; ic += ib ) {                     // 3-rd loop around micro-kernel
                ib = min(m - ic, DGEMM_MC);
                packA_mcxkc(
                        ib,
                        pb,
                        &XA[pc * lda + ic],
                        lda,
                        packA
                        );
                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA,
                        packB,
                        &C[jc * ldc + ic], 
                        ldc
                        );
            }                                                      // End 3.rd loop around micro-kernel
        }                                                          // End 4.th loop around micro-kernel
    }                                                              // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}
#endif

