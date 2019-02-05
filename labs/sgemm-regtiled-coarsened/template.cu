#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C_) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C_[(row) + (col)*m]

  // INSERT KERNEL CODE HERE
  
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  int row = bx * TILE_SZ_A + tx, col = by * TILE_SZ_B;
  float o0= 0, o1=0, o2=0, o3=0, o4=0, o5=0, o6=0, o7=0;
  float o8=0, o9=0, o10=0, o11=0, o12=0, o13=0, o14=0, o15=0;

  for (int ph = 0;ph < ceil(k * 1.0 / TILE_SZ_RATIO);ph++)
  {
    // Load from B to shared memory
    if(ph *TILE_SZ_RATIO + tx / TILE_SZ_B < k && col + tx % TILE_SZ_B < n)
      B_s[tx / TILE_SZ_B][tx % TILE_SZ_B] = B(ph * TILE_SZ_RATIO + tx / TILE_SZ_B, col + tx % TILE_SZ_B);
    else
      B_s[tx / TILE_SZ_B][tx % TILE_SZ_B] = 0;

    __syncthreads();
    
    // Compute TILE_SZ_RATIO step
    for (int step = 0;step < TILE_SZ_RATIO;step++)
    {
      if(row < m && ph * TILE_SZ_RATIO + step < k)
      {
        float A_s = A(row, ph * TILE_SZ_RATIO + step);
        o0 += A_s * B_s[step][0];
        o1 += A_s * B_s[step][1];
        o2 += A_s * B_s[step][2];
        o3 += A_s * B_s[step][3];
        o4 += A_s * B_s[step][4];
        o5 += A_s * B_s[step][5];
        o6 += A_s * B_s[step][6];
        o7 += A_s * B_s[step][7];
        o8 += A_s * B_s[step][8];
        o9 += A_s * B_s[step][9];
        o10 += A_s * B_s[step][10];
        o11 += A_s * B_s[step][11];
        o12 += A_s * B_s[step][12];
        o13 += A_s * B_s[step][13];
        o14 += A_s * B_s[step][14];
        o15 += A_s * B_s[step][15];
      } 
    }
    __syncthreads();


  }
  
  // Write back
  if(row < m)
  {
    if(by != blockDim.y)
    {
      C(row, col) = o0;
      C(row, col+1) = o1;
      C(row, col+2) = o2;
      C(row, col+3) = o3;
      C(row, col+4) = o4;
      C(row, col+5) = o5;
      C(row, col+6) = o6;
      C(row, col+7) = o7;
      C(row, col+8) = o8;
      C(row, col+9) = o9;
      C(row, col+10) = o10;
      C(row, col+11) = o11;
      C(row, col+12) = o12;
      C(row, col+13) = o13;
      C(row, col+14) = o14;
      C(row, col+15) = o15;
    }
    else
    {
      C(row, col) = o0;
      if(col + 1 < n)
        C(row, col + 1) = o1;
      if(col + 2 < n)
        C(row, col + 2) = o2;
      if(col + 3 < n)
        C(row, col + 3) = o3;
      if(col + 4 < n)
        C(row, col + 4) = o4;
      if(col + 5 < n)
        C(row, col + 5) = o5;
      if(col + 6 < n)
        C(row, col + 6) = o6;
      if(col + 7 < n)
        C(row, col + 7) = o7;
      if(col + 8 < n)
        C(row, col + 8) = o8;
      if(col + 9 < n)
        C(row, col + 9) = o9;
      if(col + 10 < n)
        C(row, col + 10) = o10;
      if(col + 11 < n)
        C(row, col + 11) = o11;
      if(col + 12 < n)
        C(row, col + 12) = o12;
      if(col + 13 < n)
        C(row, col + 13) = o13;
      if(col + 14 < n)
        C(row, col + 14) = o14;
      if(col + 15 < n)
        C(row, col + 15) = o15;
    }
  }

  #undef A
  #undef B
  #undef C

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE
    dim3 BlockSize(TILE_SZ_A, 1, 1);
    dim3 GridSize(ceil(m * 1.0 / TILE_SZ_A), ceil(n * 1.0 / TILE_SZ_B), 1);
    mysgemm<<<GridSize, BlockSize>>>(m, n, k, A, B, C);
    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

}