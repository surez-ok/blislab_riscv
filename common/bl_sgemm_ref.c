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
 * bl_sgemm_ref.c
 *
 *
 * Purpose:
 * implement reference mkl using GEMM (optional) in C.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#include <bl_sgemm.h>

void bl_sgemm_ref(
    int    m,
    int    n,
    int    k,
    float *XA,
    int    lda,
    float *XB,
    int    ldb,
    float *XC,
    int    ldc
    )
{
    // Local variables.
    int i, j, p;

    // Sanity check for early return.
    if (m == 0 || n == 0 || k == 0) {
        printf( "bl_sgemm(): early return\n" );
        return;
    }

    // Reference GEMM implementation.
#ifdef ROW_MAJOR
    assert(lda == k);
    assert(ldb == n);
    assert(ldc == n);
    // ijp 顺序
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (p = 0; p < k; p++) {
                XC[i * ldc + j] += XA[i * lda + p] * XB[p * ldb + j];
            }
        }
    }
#else /* COLUMN_MAJOR */
    assert(lda == m);
    assert(ldb == k);
    assert(ldc == m);
    // ijp 顺序
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (p = 0; p < k; p++) {
                XC[j * ldc + i] += XA[p * lda + i] * XB[j * ldb + p];
            }
        }
    }
#endif
}

