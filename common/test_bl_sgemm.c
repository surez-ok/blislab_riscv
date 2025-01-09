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
 * test_bl_sgemm.c
 *
 *
 * Purpose:
 * test driver for BLISLAB sgemm routine and reference sgemm routine.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */
#include <bl_sgemm.h>

#define TOLERANCE 1E-2
void computeError(
        int ldc,
        int ldc_ref,
        int m,
        int n,
        float *C,
        float *C_ref
    )
{
    int i, j;
    for (i = 0; i < m; i ++) {
        for ( j = 0; j < n; j ++ ) {
            if (fabs( C(i, j) - C_ref(i, j)) > TOLERANCE) {
                printf("Error: C[%d][%d] != C_ref, %E, %E\n", i, j, C(i, j), C_ref(i, j));
                break;
            }
        }
    }
}

/*
 * The timer functions are copied directly from BLIS 0.2.0
 *
 */
float bl_clock( void )
{
    static float gtod_ref_time_sec = 0.0;
    float the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (float) ts.tv_sec;
    /* in order to increase accuracy */
    norm_sec = (float) ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}

// --- Begin Linux build definitions -------------------------------------------

void test_bl_sgemm(
        int m,
        int n,
        int k
    ) 
{
    int i, j, p;
    float *A, *B, *C, *C_ref;
    float flops, ref_start, ref_time = 0.0f, bl_sgemm_start, bl_sgemm_time;
    int nrepeats;

    A = (float*)malloc(sizeof(float) * m * k);
    B = (float*)malloc(sizeof(float) * k * n);
    C = (float*)malloc(sizeof(float) * m * n);
    C_ref = (float*)malloc(sizeof(float) * m * n);

    nrepeats = 3;

    srand48(time(NULL));

#ifdef ROW_MAJOR
    int lda = k, ldb = n, ldc = n, ldc_ref = n;
#else /* COLUMN_MAJOR */
    int lda = m, ldb = k, ldc = m, ldc_ref = m;
#endif

    // Randonly generate points in [ 0, 1 ].
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            A(i, j) = (float)(drand48());
        }
    }
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
            B(i, j) = (float)(drand48());
        }
    }
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C(i, j) = (float)(0.0);
            C_ref(i, j) = (float)(0.0);
        }
    }
    
    bl_sgemm_time = 0.0f;
    for (i = 0; i < nrepeats; i++) {
        bl_sgemm_start = bl_clock();
        {
            bl_sgemm(
                    m,
                    n,
                    k,
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc
                    );
        }
        bl_sgemm_time += bl_clock() - bl_sgemm_start;
    }
    // average time
    bl_sgemm_time = bl_sgemm_time / nrepeats;

    ref_time = 0.0f; 
    for (i = 0; i < nrepeats; i++) {
        ref_start = bl_clock();
        {
            bl_sgemm_ref(
                    m,
                    n,
                    k,
                    A,
                    lda,
                    B,
                    ldb,
                    C_ref,
                    ldc_ref
                    );
        }
        ref_time += bl_clock() - ref_start;
    }
    // average time
    ref_time = ref_time / nrepeats;

    computeError(
            ldc,
            ldc_ref,
            m,
            n,
            C,
            C_ref
            );

    // Compute overall floating point operations.
    flops = (m * n / ( 1000.0 * 1000.0 )) * (2 * k);

    printf("%5d\t %5d\t %5d\t %5.2f\t %5.2f\t \n", 
            m, n, k, flops / bl_sgemm_time, flops / ref_time);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

int main(int argc, char *argv[])
{
    int m, n, k; 

    if (argc != 4) {
        printf("Usage: ./test_bl_sgemm.x M N K\n");
        printf("The example is: C(M, N) = A(M, K) x  B(K, N)\n");
        printf("Error: require 3 arguments, but only %d provided.\n", argc - 1);
        exit(0);
    }

    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &n);
    sscanf(argv[3], "%d", &k);

    test_bl_sgemm(m, n, k);

    return 0;
}

