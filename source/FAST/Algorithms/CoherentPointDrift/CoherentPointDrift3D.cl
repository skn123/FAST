__kernel void calculatePT1AndDenominator(
        __global float* fixedPoints,
        __global float* movingPoints,
        __global float* pt1,
        __global float* denominator,
        __private float c,
        __private float variance,
        __private unsigned int numMoving
        ) {
            const int n = get_global_id(0);
            const int numFixed = get_global_size(0);

            float colSum = 0.0f;
            for (int m = 0; m < numMoving; ++m) {
                float norm  = pown(fixedPoints[n]-movingPoints[m], 2)
                            + pown(fixedPoints[n+numFixed]-movingPoints[m+numMoving], 2)
                            + pown(fixedPoints[n+2*numFixed]-movingPoints[m+2*numMoving], 2);
                 colSum += exp( -norm / (2.f*variance));
            }
            colSum = max(FLT_EPSILON, colSum);
            denominator[n]  = colSum + c;
            pt1[n]          = colSum / (colSum + c);
        }

__kernel void calculateP1AndPX(
        __global float* fixedPoints,
        __global float* movingPoints,
        __global float* p1,
        __global float* denominator,
        __global float* px,
        __private float c,
        __private float variance,
        __private unsigned int numFixed
        ) {
            const int m = get_global_id(0);
            const int numMoving = get_global_size(0);

            float rowSum = 0.0f;
            float rowSumX0 = 0.0f;
            float rowSumX1 = 0.0f;
            float rowSumX2 = 0.0f;
            for (int n = 0; n < numFixed; ++n) {
                float norm  = pown(fixedPoints[n]-movingPoints[m], 2)
                            + pown(fixedPoints[n+numFixed]-movingPoints[m+numMoving], 2)
                            + pown(fixedPoints[n+2*numFixed]-movingPoints[m+2*numMoving], 2);
                float p = exp( -norm / (2.f*variance)) / denominator[n];

                rowSum   += p;
                rowSumX0 += p * fixedPoints[n];
                rowSumX1 += p * fixedPoints[n +   numFixed];
                rowSumX2 += p * fixedPoints[n + 2*numFixed];
            }

            p1[m]               = rowSum;
            px[m]               = rowSumX0;
            px[m + numMoving]   = rowSumX1;
            px[m + 2*numMoving] = rowSumX2;
        }

__kernel void calculateP1Q(
        __global float* P1,
        __global float* Q,
        __global float* P1Q
        ) {
            const int m = get_global_id(0);
            const int k = get_global_id(1);
            const int M = get_global_size(0);
            const int K = get_global_size(1);

            P1Q[m + k*M] = P1[m] * Q[m + k*M];
        }

__kernel void calculateRHS(
        __global float* P1,
        __global float* PX,
        __global float* movingPoints,
        __global float* RHS
        ) {
            const int m = get_global_id(0);
            const int d = get_global_id(1);
            const int M = get_global_size(0);
            const int D = get_global_size(1);

            float P1Y = P1[m] * movingPoints[m + d*M];
            RHS[m + d*M] = PX[m + d*M] - P1Y;
        }

__kernel void calculateGW(
        __global float* movingPoints,
        __global float* W,
        __global float* GW,
        __private float beta
        ) {
            const int m = get_global_id(0);
            const int d = get_global_id(1);
            const int M = get_global_size(0);
            const int D = get_global_size(1);

            float gw = 0.0f;
            for (int k = 0; k < M; ++k) {
                float norm = pown(movingPoints[m]-movingPoints[k], 2)
                           + pown(movingPoints[m+M]-movingPoints[k+M], 2)
                           + pown(movingPoints[m+2*M]-movingPoints[k+2*M], 2);
                gw += exp( -norm / (2.f*beta*beta) ) * W[k + d*M];
            }
            GW[m + d*M] = gw;
        }


__kernel void qr(
        __global float* a_mat,
        __global float* q_mat,
        __global float* p_mat,
        __global float* prod_mat,
        __local float* u_vec,
        __private int numCols
        ) {

            local float u_length_squared, dot;
            float prod, vec_length = 0.0f;

            int id = get_local_id(0);
            int numRows = get_global_size(0);


            // u is set to first column of A
            u_vec[id] = a_mat[id];
            barrier(CLK_LOCAL_MEM_FENCE);

            // Find first column of R (only one nonzero element)
            // and the initial reflection u0
            if(id == 0) {
                for(int i=1; i<numRows; i++) {
                    vec_length += u_vec[i] * u_vec[i];
                }
                u_length_squared = vec_length;
                vec_length = sqrt(vec_length + u_vec[0] * u_vec[0]);
                a_mat[0] = vec_length;
                u_vec[0] -= vec_length;
                u_length_squared += u_vec[0] * u_vec[0];
            }
            else {
                a_mat[id] = 0.0f;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);

            // Transform the remaining columns of A with the u0 reflection
            for(int i=1; i<numCols; i++) {
                dot = 0.0f;
                if(id == 0) {
                    for(int j=0; j<numRows; j++) {
                        dot += a_mat[i*numRows + j] * u_vec[j];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                a_mat[i*numRows + id] -= 2.f * u_vec[id] * dot / u_length_squared;
            }

            // Initialize Q matrix
            for(int i=0; i<numRows; i++) {
                q_mat[i*numRows + id] = -2.f * u_vec[i] * u_vec[id] / u_length_squared;
            }
            q_mat[id*numRows + id] += 1;
            barrier(CLK_GLOBAL_MEM_FENCE);


            // Loop through the untransformed columns.
            // For each column, calculate the reflections u and transform the columns to the right
            for(int col = 1; col < numCols-1; col++) {
                u_vec[id] = a_mat[col * numRows + id];
                barrier(CLK_LOCAL_MEM_FENCE);

                if(id == col) {
                    vec_length = 0.0f;
                    for(int i = col + 1; i < numRows; i++) {
                        vec_length += u_vec[i] * u_vec[i];
                    }
                    u_length_squared = vec_length;
                    vec_length = sqrt(vec_length + u_vec[col] * u_vec[col]);
                    u_vec[col] -= vec_length;
                    u_length_squared += u_vec[col] * u_vec[col];
                    a_mat[col * numRows + col] = vec_length;
                }
                else if(id > col) {
                    a_mat[col * numRows + id] = 0.0f;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);


                /* Transform further columns of A */
                for(int i = col+1; i < numCols; i++) {
                    if(id == 0) {
                        dot = 0.0f;
                        for(int j=col; j<numRows; j++) {
                            dot += a_mat[i*numRows + j] * u_vec[j];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(id >= col) {
                        a_mat[i*numRows + id] -= 2 * u_vec[id] * dot / u_length_squared;
                    }
                    barrier(CLK_GLOBAL_MEM_FENCE);
                }


                /* Update P matrix */
                if(id >= col) {
                    for(int i=col; i<numRows; i++) {
                        p_mat[id*numRows + i] = -2 * u_vec[i] * u_vec[id] / u_length_squared;
                    }
                    p_mat[id*numRows + id] += 1;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);


                /* Multiply q_mat * p_mat = prod_mat */
                for(int i=col; i<numRows; i++) {
                    prod = 0.0f;
                    for(int j=col; j<numRows; j++) {
                        prod += q_mat[id + j*numRows] * p_mat[i*numRows + j];
                    }
                    prod_mat[i*numRows + id] = prod;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);


                /* Place the content of prod_mat in q_mat */
                for(int i=col; i<numRows; i++) {
                    q_mat[i*numRows + id] = prod_mat[i*numRows + id];
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
            }
        }
