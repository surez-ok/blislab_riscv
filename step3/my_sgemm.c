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

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

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

// c = a*b row-major order
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
#if (defined(__riscv_vector))

  float *pInA = packA;              /* Input data matrix pointer A */
  float *pInB = packB;              /* Input data matrix pointer B */
  float *px = C;                /* Temporary output data matrix pointer */
  size_t ii, jj, kk;

  size_t l;
  vfloat32m4_t va0m4, vres0m4, vres1m4, vres2m4, vres3m4;
  vfloat32m8_t va0m8, vres0m8, vres1m8;
  /* ch = 4, mul = 4 */
   for (jj = m / 4; jj > 0; jj--) {
        px = C;
        pInB = packB;

        for (ii = n; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m4(ii);

            pInA = packA;

            vres0m4 = __riscv_vle32_v_f32m4(px, l);
            vres1m4 = __riscv_vle32_v_f32m4(px + ldc, l);
            vres2m4 = __riscv_vle32_v_f32m4(px + 2 * ldc, l);
            vres3m4 = __riscv_vle32_v_f32m4(px + 3 * ldc, l);
            for (kk = 0; kk < k; kk++) {
                va0m4 = __riscv_vle32_v_f32m4(pInB + kk * n, l); 

                vres0m4 = __riscv_vfmacc_vf_f32m4(vres0m4, *(pInA + 0), va0m4, l);
                vres1m4 = __riscv_vfmacc_vf_f32m4(vres1m4, *(pInA + k), va0m4, l);
                vres2m4 = __riscv_vfmacc_vf_f32m4(vres2m4, *(pInA + 2 * k), va0m4, l);
                vres3m4 = __riscv_vfmacc_vf_f32m4(vres3m4, *(pInA + 3 * k), va0m4, l);
                pInA++;
            }
            __riscv_vse32_v_f32m4(px, vres0m4, l);
            __riscv_vse32_v_f32m4(px + ldc, vres1m4, l);
            __riscv_vse32_v_f32m4(px + 2 * ldc, vres2m4, l);
            __riscv_vse32_v_f32m4(px + 3 * ldc, vres3m4, l);
            px += l;
            pInB += l;
        }
        packA += (k << 2);
        C += (ldc << 2);
    }
    /* ch = 2, mul = 8 */
    m = m & 3;
    for (jj = m / 2; jj > 0; jj--) {
        px = C;
        pInB = packB;

        for (ii = n; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);

            pInA = packA;

            vres0m8 = __riscv_vle32_v_f32m8(px, l);
            vres1m8 = __riscv_vle32_v_f32m8(px + ldc, l);
            for (kk = 0; kk < k; kk++) {
                va0m8 = __riscv_vle32_v_f32m8(pInB + kk * n, l);

                vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pInA + 0), va0m8, l);
                vres1m8 = __riscv_vfmacc_vf_f32m8(vres1m8, *(pInA + k), va0m8, l);
                pInA++;
            }
            __riscv_vse32_v_f32m8(px, vres0m8, l);
            __riscv_vse32_v_f32m8(px + ldc, vres1m8, l);
            px += l;
            pInB += l;
        }
        packA += (k << 1);
        C += (ldc << 1);
    }
    /* ch = 1, mul = 8 */
    m = m & 1;

    for (jj = m; jj > 0; jj--) {
        px = C;
        pInB = packB;

        for (ii = n; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);

            pInA = packA;

            vres0m8 = __riscv_vle32_v_f32m8(px, l);
            for (kk = 0; kk < k; kk++) {
                va0m8 = __riscv_vle32_v_f32m8(pInB + kk * n, l); 
                vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pInA++), va0m8, l);
            }
            __riscv_vse32_v_f32m8(px, vres0m8, l);
            px += l;
            pInB += l;
        }
        packA += (k);
        C += (ldc);
    }
#else
    int i, p, j;

    for (i = 0; i < m; i++) {
        for (p = 0; p < k; p++) {
            for (j = 0; j < n; j++) {
                C[i * ldc + j] += packA[i * k + p] * packB[p * n + j];
            }
        }  
    }
#endif
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

// c = a*b column-major order
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
#if (defined(__riscv_vector))

  float *pInA = packA;              /* Input data matrix pointer A */
  float *pInB = packB;              /* Input data matrix pointer B */
  float *px = C;                    /* Temporary output data matrix pointer */
  size_t ii, jj, kk;

  size_t l;
  vfloat32m4_t va0m4, vres0m4, vres1m4, vres2m4, vres3m4;
  vfloat32m8_t va0m8, vres0m8, vres1m8;
  /* ch = 4, mul = 4 */
   for (jj = n / 4; jj > 0; jj--) {
        px = C;
        pInA = packA;

        for (ii = m; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m4(ii);

            pInB = packB;

            vres0m4 = __riscv_vle32_v_f32m4(px, l);
            vres1m4 = __riscv_vle32_v_f32m4(px + ldc, l);
            vres2m4 = __riscv_vle32_v_f32m4(px + 2 * ldc, l);
            vres3m4 = __riscv_vle32_v_f32m4(px + 3 * ldc, l);
            for (kk = 0; kk < k; kk++) {
                va0m4 = __riscv_vle32_v_f32m4(pInA + kk * m, l);

                vres0m4 = __riscv_vfmacc_vf_f32m4(vres0m4, *(pInB + 0), va0m4, l);
                vres1m4 = __riscv_vfmacc_vf_f32m4(vres1m4, *(pInB + k), va0m4, l);
                vres2m4 = __riscv_vfmacc_vf_f32m4(vres2m4, *(pInB + 2 * k), va0m4, l);
                vres3m4 = __riscv_vfmacc_vf_f32m4(vres3m4, *(pInB + 3 * k), va0m4, l);
                pInB++;
            }
            __riscv_vse32_v_f32m4(px, vres0m4, l);
            __riscv_vse32_v_f32m4(px + ldc, vres1m4, l);
            __riscv_vse32_v_f32m4(px + 2 * ldc, vres2m4, l);
            __riscv_vse32_v_f32m4(px + 3 * ldc, vres3m4, l);
            px += l;
            pInA += l;
        }
        packB += (k << 2);
        C += (ldc << 2);
    }
    /* ch = 2, mul = 8 */
    n = n & 3;
    for (jj = n / 2; jj > 0; jj--) {
        px = C;
        pInA = packA;

        for (ii = m; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);

            pInB = packB;

            vres0m8 = __riscv_vle32_v_f32m8(px, l);
            vres1m8 = __riscv_vle32_v_f32m8(px + ldc, l);
            for (kk = 0; kk < k; kk++) {
                va0m8 = __riscv_vle32_v_f32m8(pInA + kk * m, l);

                vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pInB + 0), va0m8, l);
                vres1m8 = __riscv_vfmacc_vf_f32m8(vres1m8, *(pInB + k), va0m8, l);
                pInB++;
            }
            __riscv_vse32_v_f32m8(px, vres0m8, l);
            __riscv_vse32_v_f32m8(px + ldc, vres1m8, l);
            px += l;
            pInA += l;
        }
        packB += (k << 1);
        C += (ldc << 1);
    }
    /* ch = 1, mul = 8 */
    n = n & 1;

    for (jj = n; jj > 0; jj--) {
        px = C;
        pInA = packA;

        for (ii = m; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);

            pInB = packB;

            vres0m8 = __riscv_vle32_v_f32m8(px, l);
            for (kk = 0; kk < k; kk++) {
                va0m8 = __riscv_vle32_v_f32m8(pInA + kk * m, l); 

                vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pInB++), va0m8, l);
            }
            __riscv_vse32_v_f32m8(px, vres0m8, l);
            px += l;
            pInA += l;
        }
        packB += (k);
        C += (ldc);
    }
#else
    int i, p, j;

    for (j = 0; j < n; j++) {            // Start 2-nd loop
      for (p = 0; p < k; p++) {          // Start 1-st loop
        for (i = 0; i < m; i ++) {
           C[ j * ldc + i] += packA[p * m + i] * packB[j * k + p];
        }
      }                                  // End   1-st loop
  }                                      // 2-th loop around micro-kernel
#endif
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

