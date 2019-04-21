// haoyuan9
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
  return (x - 1) / y + 1;
}

__device__ int co_rank(int k, float *A, int m, float *B, int n) {

  int i = k < m ? k : m; // i = min(k, m)
  int j = k - i;
  int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max(0, k-n)
  int j_low = 0 > (k - m) ? 0 : k - m; // i_low = max(0, k-m)

  int delta;
  bool active = true;
  while (active) {
    // i > 0 and j < n are neglectable??
    // No, since you have to check A[i-1] and B[j-1]
    if (i > 0 && j < n && A[i - 1] > B[j]) {
      delta = ((i - i_low + 1) >> 1); // ceil((i-i_low) / 2)
      j_low = j;
      j = j + delta;
      i = i - delta;
    } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
      delta = ((j - j_low + 1) >> 1);
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      active = false;
    }
  }
  return i;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float *A, int A_len, float *B, int B_len,
                                 float *C) {
  int i = 0, j = 0, k = 0;

  while ((i < A_len) && (j < B_len)) {
    C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
  }

  if (i == A_len) {
    while (j < B_len) {
      C[k++] = B[j++];
    }
  } else {
    while (i < A_len) {
      C[k++] = A[i++];
    }
  }
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float *A, int A_len, float *B, int B_len,
                                       float *C) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  //  All lowercase letters represent thread level indices for merging.
  int k_curr = tid * ceil_div((A_len + B_len), blockDim.x * gridDim.x);
  int k_next =
      min((tid + 1) * ceil_div((A_len + B_len), blockDim.x * gridDim.x),
          A_len + B_len);
  int i_curr = co_rank(k_curr, A, A_len, B, B_len);
  int i_next = co_rank(k_next, A, A_len, B, B_len);
  int j_curr = k_curr - i_curr;
  int j_next = k_next - i_next;

  merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr,
                   &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float *A, int A_len, float *B, int B_len,
                                       float *C) {
  __shared__ float sharedAB[2 * TILE_SIZE];
  __shared__ int s1, s2;
  float *A_S = &sharedAB[0]; //
  float *B_S = &sharedAB[TILE_SIZE];

  // All capital letters represents block level indices for merging.

  // Notice the min() for C_curr.
  // This is very tricky. We also have to check if C_curr overflow.
  // For instance, (A_len + B_len) = 9095, gridDim.x = 128, then each block
  // caculates 72 outputs because ceil(9095 / 128) = 72.
  // However, 126 * 72 = 9072, and 127 * 72 = 9144. It overflows at
  // the second last block. Then for the last block, C_curr = 9144,
  // C_next = A_len + B_len = 9095. Then everything break.
  // This is because every block deal with about 0.9 more outputs.
  // Then for 127 blocks, they already produce more outputs than 9095.

  int C_curr =
      min(blockIdx.x * ceil_div(A_len + B_len, gridDim.x), A_len + B_len);
  int C_next =
      min((blockIdx.x + 1) * ceil_div(A_len + B_len, gridDim.x), A_len + B_len);

  if (threadIdx.x == 0) {
    s1 = co_rank(C_curr, A, A_len, B, B_len);
    s2 = co_rank(C_next, A, A_len, B, B_len);
  }

  __syncthreads();

  int A_curr = s1;
  int B_curr = C_curr - A_curr;
  int A_next = s2;
  int B_next = C_next - A_next;

  int counter = 0; // Iteration counter. Each iteration completes TILE_SIZE Cs
  int C_length = C_next - C_curr; // Number of Cs handled by current block
  int A_length = A_next - A_curr;
  int B_length = B_next - B_curr;

  int total_iteration = ceil_div(C_length, TILE_SIZE);
  // if (C_length == 0)
  //   total_iteration = 0;
  int C_completed = 0;
  int A_consumed = 0;
  int B_consumed = 0;

  while (counter < total_iteration) {
    __syncthreads();
    // Everytime load TILE_SIZE As and Bs into shared memory
    // Each thread loads TILE_SIZE / blockDim.x items
    for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
      // Only has  A_length - A_consumed to be merged
      if (i + threadIdx.x < min(A_length - A_consumed, TILE_SIZE))
        A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
    }
    for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
      // Same as above
      if (i + threadIdx.x < min(B_length - B_consumed, TILE_SIZE))
        B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
    }

    __syncthreads();

    // Thread level merge
    // Each block merges TILE_SIZE Cs
    // Remember there are only C_length - C_completed Cs to be merged
    int c_curr = min(min(ceil_div(TILE_SIZE, blockDim.x) * threadIdx.x,
                         C_length - C_completed),
                     TILE_SIZE);
    int c_next = min(min(ceil_div(TILE_SIZE, blockDim.x) * (threadIdx.x + 1),
                         C_length - C_completed),
                     TILE_SIZE);

    int a_curr = co_rank(c_curr, A_S, min(TILE_SIZE, A_length - A_consumed),
                         B_S, min(TILE_SIZE, B_length - B_consumed));
    int b_curr = c_curr - a_curr;
    int a_next = co_rank(c_next, A_S, min(TILE_SIZE, A_length - A_consumed),
                         B_S, min(TILE_SIZE, B_length - B_consumed));
    int b_next = c_next - a_next;

    // merge
    merge_sequential(&A_S[a_curr], a_next - a_curr, &B_S[b_curr],
                     b_next - b_curr, &C[C_curr + C_completed + c_curr]);

    // Update consumed length
    counter++;
    A_consumed =
        A_consumed + co_rank(min(TILE_SIZE, C_length - C_completed), A_S,
                             min(TILE_SIZE, A_length - A_consumed), B_S,
                             min(TILE_SIZE, B_length - B_consumed));
    C_completed = C_completed + min(TILE_SIZE, C_length - C_completed);
    B_consumed = C_completed - A_consumed;
  }
}

__device__ int co_rank_circular(int k, float *A, int m, float *B, int n,
                                int A_S_start, int B_S_start, int tile_size) {

  // These i, j, i_low, and j_low are all virtual indices of a buffer started
  // from 0. The actual buffer starts from A_S_start and B_S_start
  int i = k < m ? k : m; // i = min(k, m)
  int j = k - i;
  int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max(0, k-n)
  int j_low = 0 > (k - m) ? 0 : k - m; // i_low = max(0, k-m)

  int delta;
  bool active = true;
  while (active) {

    // Convert i, j, i - 1, j - 1 into actual indices
    int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size
                                             : A_S_start + i;
    int i_m_1_cir = (A_S_start + i - 1 >= tile_size)
                        ? A_S_start + i - 1 - tile_size
                        : A_S_start + i - 1;

    int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size
                                             : B_S_start + j;
    int j_m_1_cir = (B_S_start + j - 1 >= tile_size)
                        ? B_S_start + j - 1 - tile_size
                        : B_S_start + j - 1;

    // i > 0 and j < n are neglectable??
    // No, since you have to check A[i-1] and B[j-1]
    if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
      delta = ((i - i_low + 1) >> 1); // ceil((i-i_low) / 2)
      j_low = j;
      j = j + delta;
      i = i - delta;
    } else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
      delta = ((j - j_low + 1) >> 1);
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      active = false;
    }
  }
  return i;
}

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential_circular(float *A, int A_len, float *B,
                                          int B_len, float *C, int A_S_start,
                                          int B_S_start, int tile_size) {
  int i = 0, j = 0, k = 0;

  while ((i < A_len) && (j < B_len)) {
    // Convert virtual indices into actual indices
    int i_cir =
        (A_S_start + i > tile_size) ? A_S_start + i - tile_size : A_S_start + i;
    int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size
                                             : B_S_start + j;
    if (A[i_cir] <= B[j_cir]) {
      C[k++] = A[i_cir];
      i++;
    } else {
      C[k++] = B[j_cir];
      j++;
    }
  }

  if (i == A_len) {
    while (j < B_len) {
      int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size
                                               : B_S_start + j;
      C[k++] = B[j_cir];
      j++;
    }
  } else {
    while (i < A_len) {
      int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size
                                               : A_S_start + i;
      C[k++] = A[i_cir];
      i++;
    }
  }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float *A, int A_len, float *B,
                                                 int B_len, float *C) {

  __shared__ float sharedAB[2 * TILE_SIZE];
  __shared__ int s1, s2;
  float *A_S = &sharedAB[0]; //
  float *B_S = &sharedAB[TILE_SIZE];

  // All capital letters represents block level indices for merging.

  // Notice the min() for C_curr.
  // This is very tricky. We also have to check if C_curr overflow.
  // For instance, (A_len + B_len) = 9095, gridDim.x = 128, then each block
  // caculates 72 outputs because ceil(9095 / 128) = 72.
  // However, 126 * 72 = 9072, and 127 * 72 = 9144. It overflows at
  // the second last block. Then for the last block, C_curr = 9144,
  // C_next = A_len + B_len = 9095. Then everything break.
  // This is because every block deal with about 0.9 more outputs.
  // Then for 127 blocks, they already produce more outputs than 9095.

  int C_curr =
      min(blockIdx.x * ceil_div(A_len + B_len, gridDim.x), A_len + B_len);
  int C_next =
      min((blockIdx.x + 1) * ceil_div(A_len + B_len, gridDim.x), A_len + B_len);

  if (threadIdx.x == 0) {
    s1 = co_rank(C_curr, A, A_len, B, B_len);
    s2 = co_rank(C_next, A, A_len, B, B_len);
  }

  __syncthreads();

  int A_curr = s1;
  int B_curr = C_curr - A_curr;
  int A_next = s2;
  int B_next = C_next - A_next;

  int counter = 0; // Iteration counter. Each iteration completes TILE_SIZE Cs
  int C_length = C_next - C_curr; // Number of Cs handled by current block
  int A_length = A_next - A_curr;
  int B_length = B_next - B_curr;

  int total_iteration = ceil_div(C_length, TILE_SIZE);

  int C_completed = 0;
  int A_consumed = 0;
  int B_consumed = 0;

  int A_S_start = 0;
  int B_S_start = 0;
  int A_S_consumed = TILE_SIZE;
  int B_S_consumed = TILE_SIZE;

  while (counter < total_iteration) {
    __syncthreads();
    // Everytime load TILE_SIZE As and Bs into shared memory
    // Each thread loads TILE_SIZE / blockDim.x items

    for (int i = 0; i < A_S_consumed; i += blockDim.x) {
      // Only has  A_length - A_consumed to be merged
      if (i + threadIdx.x <
          min(A_length - A_consumed -
                  min((TILE_SIZE - A_S_consumed), A_length - A_consumed),
              A_S_consumed)) {
        A_S[(A_S_start + (TILE_SIZE - A_S_consumed) + i + threadIdx.x) %
            TILE_SIZE] = A[A_curr + A_consumed + i + threadIdx.x +
                           (TILE_SIZE - A_S_consumed)];
      }
    }
    for (int i = 0; i < B_S_consumed; i += blockDim.x) {
      // Same as above
      if (i + threadIdx.x <
          min(B_length - B_consumed -
                  min((TILE_SIZE - B_S_consumed), B_length - B_consumed),
              B_S_consumed)) {
        B_S[(B_S_start + (TILE_SIZE - B_S_consumed) + i + threadIdx.x) %
            TILE_SIZE] = B[B_curr + B_consumed + i + threadIdx.x +
                           (TILE_SIZE - B_S_consumed)];
      }
    }

    __syncthreads();

    // Thread level merge
    // Each block merges TILE_SIZE Cs
    // Remember there are only C_length - C_completed Cs to be merged
    int c_curr = min(min(ceil_div(TILE_SIZE, blockDim.x) * threadIdx.x,
                         C_length - C_completed),
                     TILE_SIZE);
    int c_next = min(min(ceil_div(TILE_SIZE, blockDim.x) * (threadIdx.x + 1),
                         C_length - C_completed),
                     TILE_SIZE);

    int a_curr = co_rank_circular(
        c_curr, A_S, min(TILE_SIZE, A_length - A_consumed), B_S,
        min(TILE_SIZE, B_length - B_consumed), A_S_start, B_S_start, TILE_SIZE);
    int b_curr = c_curr - a_curr;
    int a_next = co_rank_circular(
        c_next, A_S, min(TILE_SIZE, A_length - A_consumed), B_S,
        min(TILE_SIZE, B_length - B_consumed), A_S_start, B_S_start, TILE_SIZE);
    int b_next = c_next - a_next;

    // merge
    merge_sequential_circular(A_S, a_next - a_curr, B_S, b_next - b_curr,
                              C + C_curr + C_completed + c_curr,
                              A_S_start + a_curr, B_S_start + b_curr,
                              TILE_SIZE);

    // Update consumed length
    counter++;
    A_S_consumed = co_rank_circular(min(TILE_SIZE, C_length - C_completed), A_S,
                                    min(TILE_SIZE, A_length - A_consumed), B_S,
                                    min(TILE_SIZE, B_length - B_consumed),
                                    A_S_start, B_S_start, TILE_SIZE);

    B_S_consumed = min(TILE_SIZE, C_length - C_completed) - A_S_consumed;
    A_consumed += A_S_consumed;
    C_completed += min(TILE_SIZE, C_length - C_completed);
    B_consumed = C_completed - A_consumed;

    A_S_start += A_S_consumed;
    if (A_S_start >= TILE_SIZE)
      A_S_start = A_S_start - TILE_SIZE;
    B_S_start += B_S_consumed;
    if (B_S_start >= TILE_SIZE)
      B_S_start = B_S_start - TILE_SIZE;
  }
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float *A, int A_len, float *B, int B_len, float *C) {
  const int numBlocks = 128;
  gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float *A, int A_len, float *B, int B_len, float *C) {
  const int numBlocks = 128;
  gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float *A, int A_len, float *B, int B_len,
                               float *C) {
  const int numBlocks = 128;
  gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B,
                                                              B_len, C);
}
