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

#include <stdio.h>

#include "bl_sgemm.h"
#include "bl_config.h"

inline void packA_mcxkc(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        float *packA
        )
{
    int    i, p;

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            *packA ++ = *(XA + p * ldXA + i);
        }
    }
}

/*
 * --------------------------------------------------------------------------
 */

inline void packB_kcxnc(
        int    n,
        int    k,
        float *XB,
        int    ldXB,
        float *packB
        )
{
    int    j, p;

    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            *packB ++ = *(XB + j * ldXB + p);
        }
    }
}

#if (defined(__riscv_vector))
#include <riscv_vector.h>
int sgemm_mat_opt_macc(int bm, int bn, int bk, float* ba, float* bb, float* c, int ldc)
{
  float *pInA = ba;              /* Input data matrix pointer A */
  float *pInB = bb;              /* Input data matrix pointer B */
  float *px = c;                /* Temporary output data matrix pointer */
  int ii, jj, kk;

  size_t l;
  vfloat32m4_t va0m4, vres0m4, vres1m4, vres2m4, vres3m4;
  vfloat32m8_t va0m8, vres0m8, vres1m8;
  /* ch = 4, mul = 4 */
   for (jj = bn / 4; jj > 0; jj--) {
        px = c;
        pInA = ba;

        for (ii = bm; ii > 0; ii -= l) {
            l = vsetvl_e32m4(ii);

            pInB = bb;

            vres0m4 = vfmv_v_f_f32m4(0.0, l);
            vres1m4 = vmv_v_v_f32m4(vres0m4, l);
            vres2m4 = vmv_v_v_f32m4(vres0m4, l);
            vres3m4 = vmv_v_v_f32m4(vres0m4, l);
            for (kk = bk; kk > 0; kk--) {
                va0m4 = vle32_v_f32m4(pInA, l); 

                vres0m4 = vfmacc_vf_f32m4(vres0m4, *(pInB + 0), va0m4, l);
                vres1m4 = vfmacc_vf_f32m4(vres1m4, *(pInB + bk), va0m4, l);
                vres2m4 = vfmacc_vf_f32m4(vres2m4, *(pInB + 2 * bk), va0m4, l);
                vres3m4 = vfmacc_vf_f32m4(vres3m4, *(pInB + 3 * bk), va0m4, l);
                pInA += bm;
                pInB += 1;
            }
            va0m4 = vle32_v_f32m4(px, l);
            va0m4 = vfadd_vv_f32m4(va0m4, vres0m4, l);
            vse32_v_f32m4(px, va0m4, l);
            va0m4 = vle32_v_f32m4(px + ldc, l);
            va0m4 = vfadd_vv_f32m4(va0m4, vres1m4, l);
            vse32_v_f32m4(px + ldc, va0m4, l);
            va0m4 = vle32_v_f32m4(px + 2 * ldc, l);
            va0m4 = vfadd_vv_f32m4(va0m4, vres2m4, l);
            vse32_v_f32m4(px + 2 * ldc, va0m4, l);
            va0m4 = vle32_v_f32m4(px + 3 *ldc, l);
            va0m4 = vfadd_vv_f32m4(va0m4, vres3m4, l);
            vse32_v_f32m4(px + 3 * ldc, va0m4, l);
            px += l;
            pInA = ba + bm - ii + l;
        }
        bb += (bk << 2);
        c += (ldc << 2);
    }
    /* ch = 2, mul = 8 */
    bn = bn & 3;
    for (jj = bn / 2; jj > 0; jj--) {
        px = c;
        pInA = ba;

        for (ii = bm; ii > 0; ii -= l) {
            l = vsetvl_e32m8(ii);

            pInB = bb;

            vres0m8 = vfmv_v_f_f32m8(0.0, l);
            vres1m8 = vmv_v_v_f32m8(vres0m8, l);
            for (kk = bk; kk > 0; kk--) {
                va0m8 = vle32_v_f32m8(pInA, l); 

                vres0m8 = vfmacc_vf_f32m8(vres0m8, *(pInB + 0), va0m8, l);
                vres1m8 = vfmacc_vf_f32m8(vres1m8, *(pInB + bk), va0m8, l);
                pInA += bm;
                pInB += 1;
            }
            va0m8 = vle32_v_f32m8(px, l);
            va0m8 = vfadd_vv_f32m8(va0m8, vres0m8, l);
            vse32_v_f32m8(px, va0m8, l);
            va0m8 = vle32_v_f32m8(px + ldc, l);
            va0m8 = vfadd_vv_f32m8(va0m8, vres1m8, l);
            vse32_v_f32m8(px + ldc, va0m8, l);
            px += l;
            pInA = ba + bm - ii + l;
        }
        bb += (bk << 1);
        c += (ldc << 1);
    }
    /* ch = 1, mul = 8 */
    bn = bn & 1;

    for (jj = bn; jj > 0; jj--) {
        px = c;
        pInA = ba;

        for (ii = bm; ii > 0; ii -= l) {
            l = vsetvl_e32m8(ii);

            pInB = bb;

            vres0m8 = vfmv_v_f_f32m8(0.0, l);
            for (kk = bk; kk > 0; kk--) {
                va0m8 = vle32_v_f32m8(pInA, l); 

                vres0m8 = vfmacc_vf_f32m8(vres0m8, *(pInB + 0), va0m8, l);
                pInA += bm;
                pInB += 1;
            }
            va0m8 = vle32_v_f32m8(px, l);
            va0m8 = vfadd_vv_f32m8(va0m8, vres0m8, l);
            vse32_v_f32m8(px, va0m8, l);
            px += l;
            pInA = ba + bm - ii + l;
        }
        bb += (bk);
        c += (ldc);
    }
  return 0;
}
#endif
/*
 * --------------------------------------------------------------------------
 */
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
#if !(defined(__riscv_vector))
    int    i, p, j;

    for ( j = 0; j < n; j ++ ) {            // Start 2-nd loop
      for ( p = 0; p < k; p ++ ) {          // Start 1-st loop
          float *p_elemC = &(C[ j * ldc]);
          for ( i = 0; i < m; i ++ ) {      // Start 0-th loop
              C[ j * ldc + i] += packA[ p * m + i] * packB[ j * k + p ];
          }                                 // End   0-th loop
      }                                     // End   1-st loop
  }                                         // 2-th loop around micro-kernel
#else
    sgemm_mat_opt_macc(m, n, k, packA, packB, C, ldc);
#endif                                      
}

// C must be aligned
void bl_sgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        )
{
    int    i, j, p;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    float *packA, *packB;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_sgemm(): early return\n" );
        return;
    }

    // Allocate packing buffers
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ), sizeof(float) );
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 ), sizeof(float) );

    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                 // 5-th loop around micro-kernel
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += pb ) {                                   // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );

            packB_kcxnc(
                    jb,
                    pb,
                    &XB[ jc * ldb +  pc],
                    ldb,
                    packB
                    );

            for ( ic = 0; ic < m; ic += ib ) {                               // 3-rd loop around micro-kernel

                ib = min( m - ic, DGEMM_MC );

                packA_mcxkc(
                        ib,
                        pb,
                        &XA[ pc * lda + ic],
                        lda,
                        packA
                        );
                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA,
                        packB,
                        &C[ jc * ldc + ic ], 
                        ldc
                        );
            }                                                                     // End 3.rd loop around micro-kernel
        }                                                                         // End 4.th loop around micro-kernel
    }                                                                             // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}

