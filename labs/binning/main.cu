#include "helper.hpp"


/******************************************************************************
 GPU main computation kernels
*******************************************************************************/

__global__ void gpu_normal_kernel(float *in_val, float *in_pos, float *out,
                                  int grid_size, int num_in) {
  //@@ INSERT CODE HERE

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float result = 0;
  if (idx < grid_size){
    for (int i = 0;i < num_in;i++){
      const float dist = in_pos[i] - idx;
      result += in_val[i] * in_val[i] / (dist * dist);
    }
    out[idx] = result;
  }

}

__global__ void gpu_cutoff_kernel(float *in_val, float *in_pos, float *out,
                                  int grid_size, int num_in,
                                  float cutoff2) {
  //@@ INSERT CODE HERE
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float result = 0;
  if (idx < grid_size){
    for (int i =  0;i < num_in;i++){
      const float dist = in_pos[i] - idx;
      const float dist2 = dist * dist;
      if (dist2 >= cutoff2)                                                                                                                        
        continue;
      result += in_val[i] * in_val[i] / dist2;                                            
    }
    out[idx] = result;
  }
}

// Ver 1.0
// __global__ void gpu_cutoff_binned_kernel(int *bin_ptrs,
//                                          float *in_val_sorted,
//                                          float *in_pos_sorted, float *out,
//                                          int grid_size, float cutoff) {

// //@@ INSERT CODE HERE
// #define BLOCK_SIZE 512
// const int tx = threadIdx.x, bx = blockIdx.x, bs = blockDim.x;
// const int idx = tx + bx * bs;
// if (idx < grid_size){
//   const int cutoff2 = cutoff * cutoff;
//   float result = 0;
//   // Calculate neighborhood offset for one thread(one lattice point)
//   const int max_thread_binIdx = idx + cutoff < grid_size ? (int) ((idx + cutoff) / grid_size * NUM_BINS) : NUM_BINS-1;
//   const int min_thread_binIdx = idx - cutoff >= 0 ? (int) ((idx - cutoff) / grid_size * NUM_BINS) : 0;
//   const int max_thread_inIdx = bin_ptrs[max_thread_binIdx + 1];
//   const int min_thread_inIdx = bin_ptrs[min_thread_binIdx];
  
//   // Compute
//   for (int j = min_thread_inIdx;j < max_thread_inIdx;j++){
//     const float dist2 = (idx - in_pos_sorted[j]) * (idx - in_pos_sorted[j]);
//     if (dist2 <= cutoff2)
//       result += in_val_sorted[j] * in_val_sorted[j] / dist2;
//   }
  
//   out[idx] = result;
// }
// #undef BLOCK_SIZE
// }

// Ver 2.0
// __global__ void gpu_cutoff_binned_kernel(int *bin_ptrs,
//                                          float *in_val_sorted,
//                                          float *in_pos_sorted, float *out,
//                                          int grid_size, float cutoff) {

// //@@ INSERT CODE HERE
// #define BLOCK_SIZE 512
// __shared__ float shared_pos[BLOCK_SIZE];
// __shared__ float shared_val[BLOCK_SIZE];
// const int tx = threadIdx.x, bx = blockIdx.x, bs = blockDim.x;
// const int idx = tx + bx * bs;
// const int cutoff2 = cutoff * cutoff;
// float result = 0;
// // Locate possible bin index, same as neighborhood offset list in 2D/3D 
// const float upper = (bx + 1) * bs + cutoff - 1;
// const float lower = bx * bs - cutoff;
// const int max_binIdx = upper < grid_size ? (int) (upper / grid_size * NUM_BINS) : NUM_BINS-1;                                        
// const int min_binIdx = lower >= 0 ? (int) (lower / grid_size * NUM_BINS) : 0;                                                                                                                 
// const int max_inIdx = bin_ptrs[max_binIdx + 1];
// const int min_inIdx = bin_ptrs[min_binIdx];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

// // Collaboratively load one valid bins into shared memory and compute result correspondingly
// for (int i = min_inIdx;i < max_inIdx;i += bs){

//   // Load
//   if (tx + i < max_inIdx){
//     shared_pos[tx] = in_pos_sorted[tx + i];
//     shared_val[tx] = in_val_sorted[tx + i];
//   }
//   else{
//     shared_pos[tx] = -1;
//     shared_val[tx] = 0;
//   }

//   __syncthreads();

//   // Compute
//   for (int j = 0;j < bs;j++){
//     const float dist2 = (idx - shared_pos[j]) * (idx - shared_pos[j]);
//     if (dist2 <= cutoff2)
//       result += shared_val[j] * shared_val[j] / dist2;
//   }
//   __syncthreads();
// }
// if (idx < grid_size)
//   out[idx] = result;
  
// #undef BLOCK_SIZE
// }

// Ver 3.0
__global__ void gpu_cutoff_binned_kernel(int *bin_ptrs,
                                         float *in_val_sorted,
                                         float *in_pos_sorted, float *out,
                                         int grid_size, float cutoff) {

//@@ INSERT CODE HERE
#define BLOCK_SIZE 512
__shared__ float shared_pos[BLOCK_SIZE];
__shared__ float shared_val[BLOCK_SIZE];
const int tx = threadIdx.x, bx = blockIdx.x, bs = blockDim.x;
const int idx = tx + bx * bs;
const int cutoff2 = cutoff * cutoff;
float result = 0;
// Locate possible bin index, same as neighborhood offset list in 2D/3D 
const float upper = (bx + 1) * bs + cutoff - 1;
const float lower = bx * bs - cutoff;
const int max_binIdx = upper < grid_size ? (int) (upper / grid_size * NUM_BINS) : NUM_BINS-1;                                        
const int min_binIdx = lower >= 0 ? (int) (lower / grid_size * NUM_BINS) : 0;
const int max_inIdx = bin_ptrs[max_binIdx + 1];
const int min_inIdx = bin_ptrs[min_binIdx];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

// Calculate neighborhood offset for one thread
const int max_thread_binIdx = idx + cutoff < grid_size ? (int) ((idx + cutoff) / grid_size * NUM_BINS) : NUM_BINS-1;
const int min_thread_binIdx = idx - cutoff >= 0 ? (int) ((idx - cutoff) / grid_size * NUM_BINS) : 0;
const int max_thread_inIdx = bin_ptrs[max_thread_binIdx + 1];
const int min_thread_inIdx = bin_ptrs[min_thread_binIdx];
// Collaboratively load one valid bins into shared memory and compute result correspondingly
for (int i = min_inIdx;i < max_inIdx;i += bs){

  // Load
  if (tx + i < max_inIdx){
    shared_pos[tx] = in_pos_sorted[tx + i];
    shared_val[tx] = in_val_sorted[tx + i];
  }
  else{
    shared_pos[tx] = -1;
    shared_val[tx] = 0;
  }

  __syncthreads();

  // Compute
  // Calculate the index range each thread(grid point) needs to compute
  // This version is much faster than the one above, it reduces many unnaccessary computation.
  const int start = max(min_thread_inIdx - i, 0);
  const int end = max(0, min(max_thread_inIdx - i, BLOCK_SIZE));
    for (int j = start;j < end;j++){
    const float dist2 = (idx - shared_pos[j]) * (idx - shared_pos[j]);
    if (dist2 <= cutoff2)
      result += shared_val[j] * shared_val[j] / dist2;
  }
  __syncthreads();
}
if (idx < grid_size)
  out[idx] = result;
  
#undef BLOCK_SIZE
}
/******************************************************************************
 Main computation functions
*******************************************************************************/

void cpu_normal(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in) {

  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const float in_val2 = in_val[inIdx] * in_val[inIdx];
    for (int outIdx = 0; outIdx < grid_size; ++outIdx) {
      const float dist = in_pos[inIdx] - (float)outIdx;
      out[outIdx] += in_val2 / (dist * dist);
    }
  }
}

void gpu_normal(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_normal_kernel<<<numBlocks, numThreadsPerBlock>>>(in_val, in_pos, out,
                                                       grid_size, num_in);
}

void gpu_cutoff(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in, float cutoff2) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_cutoff_kernel<<<numBlocks, numThreadsPerBlock>>>(
      in_val, in_pos, out, grid_size, num_in, cutoff2);
}

void gpu_cutoff_binned(int *bin_ptrs, float *in_val_sorted,
                       float *in_pos_sorted, float *out, int grid_size,
                       float cutoff) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_cutoff_binned_kernel<<<numBlocks, numThreadsPerBlock>>>(
      bin_ptrs, in_val_sorted, in_pos_sorted, out, grid_size, cutoff);
}

/******************************************************************************
 Preprocessing kernels
*******************************************************************************/
// Ver 1.0
// __global__ void histogram(float *in_pos, int *bin_counts, int num_in,
//                           int grid_size) {

//   //@@ INSERT CODE HERE
//   // NUM_BINS = 1024  BLOCK_SIZE = 512
//   const int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx < num_in){
//     const int binIdx = (int) (in_pos[idx] / grid_size * NUM_BINS);
//     atomicAdd(&(bin_counts[binIdx]), 1);
//   }
// }

// Ver 2.0
__global__ void histogram(float *in_pos, int *bin_counts, int num_in,
                          int grid_size) {

  //@@ INSERT CODE HERE
  // NUM_BINS = 1024  BLOCK_SIZE = 512
  __shared__ float share_bin_counts[NUM_BINS];

  const int tx = threadIdx.x;
  const int bs = blockDim.x;
  const int idx = tx + blockIdx.x * bs;

  // Is it faster? 
  // Use shared memory reduce conflicts, but the global memory access per block
  // increases from 512 to 1024. We also have to add two sycthreads().
  // For some test cases, num_in / grid_size = 40, which is a large number,
  // conflict accesses will be frequent. It might be a good idea to use shared memory
  // But for some small num_in / grid_size, it is definitely not worth it.
  
  // Initialize local hist
  share_bin_counts[tx] = 0;
  share_bin_counts[tx + bs] = 0;
  __syncthreads();

  if (idx < num_in){
    const int binIdx = (int) ((in_pos[idx] / grid_size) * NUM_BINS);
    atomicAdd(&(share_bin_counts[binIdx]), 1);
  }
  __syncthreads();


  atomicAdd(&(bin_counts[tx]), share_bin_counts[tx]);
  atomicAdd(&(bin_counts[tx + bs]), share_bin_counts[tx + bs]);


}


// __global__ void scan(int *bin_counts, int *bin_ptrs) {

//   //@@ INSERT CODE HERE
//   // Load the input into shared memory
//   #define BLOCK_SIZE 512
//   __shared__ float array[2 * BLOCK_SIZE];
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int start = bid * BLOCK_SIZE * 2 + tid;
//   if(start < NUM_BINS)
//     array[tid] = bin_counts[start];
//   else
//     array[tid] = 0;
//   if(start + BLOCK_SIZE < NUM_BINS)
//     array[tid + BLOCK_SIZE] = bin_counts[start + BLOCK_SIZE];
//   else
//     array[tid + BLOCK_SIZE] = 0;
  
//   // Reduction phase
//   int stride = 1;
//   while(stride < 2 * BLOCK_SIZE)
//   {
//     __syncthreads();
//     int index = (tid + 1) * stride * 2 - 1;
//     if(index < 2 * BLOCK_SIZE)
//       array[index] += array[index - stride];
//     stride *= 2;
//   }
  
//   // Post scan phase
//   stride = BLOCK_SIZE / 2;
//   while(stride > 0)
//   {
//     __syncthreads();
//     int index = (tid + 1) * 2 * stride - 1;
//     if(index + stride < 2 * BLOCK_SIZE)
//       array[index + stride] += array[index];
//     stride /= 2;
//   }
  
//   __syncthreads();
//   // Directly write output
//   if(start < NUM_BINS)
//     bin_ptrs[start + 1] = array[tid];
//   if(start + BLOCK_SIZE < NUM_BINS)
//     bin_ptrs[start + BLOCK_SIZE + 1] = array[tid + BLOCK_SIZE];
//   // Since the number of bins is 1024, which just fits into one block
//   // No need for futher prefixFixup
//   if(tid == 0)
//     bin_ptrs[0] = 0;
//   #undef BLOCK_SIZE
// }


__global__ void scan(int *bin_counts, int *bin_ptrs) {

  //@@ INSERT CODE HERE
  // Load the input into shared memory
  #define BLOCK_SIZE 512
  __shared__ float array[2 * BLOCK_SIZE];
  int tx = threadIdx.x;
  array[tx] = bin_counts[tx];
  array[tx + BLOCK_SIZE] = bin_counts[tx + BLOCK_SIZE];
  
  // Reduction phase
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE)
  {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE)
      array[index] += array[index - stride];
    stride *= 2;
  }
  
  // Post scan phase
  stride = BLOCK_SIZE / 2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (tx + 1) * 2 * stride - 1;
    if(index + stride < 2 * BLOCK_SIZE)
      array[index + stride] += array[index];
    stride /= 2;
  }
  
  __syncthreads();
  // Directly write output
  bin_ptrs[tx + 1] = array[tx];
  bin_ptrs[tx + BLOCK_SIZE + 1] = array[tx + BLOCK_SIZE];
  // Since the number of bins is 1024, which just fits into one block
  // No need for futher prefixFixup
  if(tx == 0)
    bin_ptrs[0] = 0;
  #undef BLOCK_SIZE
}


__global__ void sort(float *in_val, float *in_pos, float *in_val_sorted,
                     float *in_pos_sorted, int grid_size, int num_in,
                     int *bin_counts, int *bin_ptrs) {

  //@@ INSERT CODE HERE
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_in){
    const int binIdx = (int) (in_pos[idx] / grid_size * NUM_BINS);
    const int count = atomicAdd(&(bin_counts[binIdx]), -1);
    const int newIdx = bin_ptrs[binIdx + 1] - count;
    in_pos_sorted[newIdx] = in_pos[idx];
    in_val_sorted[newIdx] = in_val[idx];
  }
}

/******************************************************************************
 Preprocessing functions
*******************************************************************************/

static void cpu_preprocess(float *in_val, float *in_pos,
                           float *in_val_sorted, float *in_pos_sorted,
                           int grid_size, int num_in, int *bin_counts,
                           int *bin_ptrs) {

  // Histogram the input positions
  for (int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
    bin_counts[binIdx] = 0;
  }
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
    ++bin_counts[binIdx];
  }

  // Scan the histogram to get the bin pointers
  bin_ptrs[0] = 0;
  for (int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
    bin_ptrs[binIdx + 1] = bin_ptrs[binIdx] + bin_counts[binIdx];
  }

  // Sort the inputs into the bins
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
    const int newIdx = bin_ptrs[binIdx + 1] - bin_counts[binIdx];
    --bin_counts[binIdx];
    in_val_sorted[newIdx] = in_val[inIdx];
    in_pos_sorted[newIdx] = in_pos[inIdx];
  }
}

static void gpu_preprocess(float *in_val, float *in_pos,
                           float *in_val_sorted, float *in_pos_sorted,
                           int grid_size, int num_in, int *bin_counts,
                           int *bin_ptrs) {

  const int numThreadsPerBlock = 512;

  // Histogram the input positions
  // ceil(num_in / numThreadsPerBlock)
  
  histogram<<<(num_in - 1) / numThreadsPerBlock + 1, numThreadsPerBlock>>>(in_pos, bin_counts, num_in,
                                        grid_size);

  // Scan the histogram to get the bin pointers
  if (NUM_BINS != 1024) {
    FAIL("NUM_BINS must be 1024. Do not change.");
    return;
  }
  scan<<<1, numThreadsPerBlock>>>(bin_counts, bin_ptrs);

  // Sort the inputs into the bins
  sort<<<(num_in -1) / numThreadsPerBlock + 1, numThreadsPerBlock>>>(in_val, in_pos, in_val_sorted,
                                   in_pos_sorted, grid_size, num_in,
                                   bin_counts, bin_ptrs);
}


template <Mode mode>
int eval(const int num_in, const int max, const int grid_size) {
  const std::string mode_info = mode_name(mode);
  const std::string conf_info =
      std::string("[len:") + std::to_string(num_in) + "/max:" + std::to_string(max) + "/gridSize:" + std::to_string(grid_size) + "]";

  // Initialize host variables
  // ----------------------------------------------

  // Variables
  std::vector<float> in_val_h;
  std::vector<float> in_pos_h;
  float *in_val_d = nullptr;
  float *in_pos_d = nullptr;
  float *out_d    = nullptr;

  // Constants
  const float cutoff  = 3000.0f; // Cutoff distance for optimized computation
  const float cutoff2 = cutoff * cutoff;

  // Extras needed for input binning
  std::vector<int> bin_counts_h;
  std::vector<int> bin_ptrs_h;
  std::vector<float> in_val_sorted_h;
  std::vector<float> in_pos_sorted_h;
  int *bin_counts_d      = nullptr;
  int *bin_ptrs_d        = nullptr;
  float *in_val_sorted_d = nullptr;
  float *in_pos_sorted_d = nullptr;

  in_val_h = generate_input(num_in, max);
  in_pos_h = generate_input(num_in, grid_size);

  std::vector<float> out_h(grid_size);
  std::fill_n(out_h.begin(), grid_size, 0.0f);

  INFO("Running " << mode_info << conf_info);

  // CPU Preprocessing
  // ------------------------------------------------------

  if (mode == Mode::GPUBinnedCPUPreprocessing) {

    timer_start("Allocating data for preprocessing");
    // Data structures needed to preprocess the bins on the CPU
    bin_counts_h.reserve(NUM_BINS);
    bin_ptrs_h.reserve(NUM_BINS + 1);
    in_val_sorted_h.reserve(num_in);
    in_pos_sorted_h.reserve(num_in);

    cpu_preprocess(in_val_h.data(), in_pos_h.data(), in_val_sorted_h.data(), in_pos_sorted_h.data(), grid_size, num_in, bin_counts_h.data(),
                   bin_ptrs_h.data());
    timer_stop();
  }
  // Allocate device variables
  // ----------------------------------------------

  if (mode != Mode::CPUNormal) {

    timer_start("Allocating data");
    // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
    if (mode != Mode::GPUBinnedCPUPreprocessing) {
      THROW_IF_ERROR(cudaMalloc((void **) &in_val_d, num_in * sizeof(float)));
      THROW_IF_ERROR(cudaMalloc((void **) &in_pos_d, num_in * sizeof(float)));
    }

    // All modes need the output array
    THROW_IF_ERROR(cudaMalloc((void **) &out_d, grid_size * sizeof(float)));

    // Only binning modes need binning information
    if (mode == Mode::GPUBinnedCPUPreprocessing || mode == Mode::GPUBinnedGPUPreprocessing) {
      THROW_IF_ERROR(cudaMalloc((void **) &in_val_sorted_d, num_in * sizeof(float)));
      THROW_IF_ERROR(cudaMalloc((void **) &in_pos_sorted_d, num_in * sizeof(float)));
      THROW_IF_ERROR(cudaMalloc((void **) &bin_ptrs_d, (NUM_BINS + 1) * sizeof(int)));

      if (mode == Mode::GPUBinnedGPUPreprocessing) {
        // Only used in preprocessing but not the actual computation
        THROW_IF_ERROR(cudaMalloc((void **) &bin_counts_d, NUM_BINS * sizeof(int)));
      }
    }

    cudaDeviceSynchronize();
    timer_stop();
  }

  // Copy host variables to device
  // ------------------------------------------

  if (mode != Mode::CPUNormal) {
    timer_start("Copying data");
    // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
    if (mode != Mode::GPUBinnedCPUPreprocessing) {
      THROW_IF_ERROR(cudaMemcpy(in_val_d, in_val_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
      THROW_IF_ERROR(cudaMemcpy(in_pos_d, in_pos_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
    }

    // All modes need the output array
    THROW_IF_ERROR(cudaMemset(out_d, 0, grid_size * sizeof(float)));

    if (mode == Mode::GPUBinnedCPUPreprocessing) {
      THROW_IF_ERROR(cudaMemcpy(in_val_sorted_d, in_val_sorted_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
      THROW_IF_ERROR(cudaMemcpy(in_pos_sorted_d, in_pos_sorted_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
      THROW_IF_ERROR(cudaMemcpy(bin_ptrs_d, bin_ptrs_h.data(), (NUM_BINS + 1) * sizeof(int), cudaMemcpyHostToDevice));
    } else if (mode == Mode::GPUBinnedGPUPreprocessing) {
      // If preprocessing on the GPU, bin counts need to be initialized
      //  and nothing needs to be copied
      THROW_IF_ERROR(cudaMemset(bin_counts_d, 0, NUM_BINS * sizeof(int)));
    }

    THROW_IF_ERROR(cudaDeviceSynchronize());
    timer_stop();
  }

  // GPU Preprocessing
  // ------------------------------------------------------

  if (mode == Mode::GPUBinnedGPUPreprocessing) {

    timer_start("Preprocessing data on the GPU...");

    gpu_preprocess(in_val_d, in_pos_d, in_val_sorted_d, in_pos_sorted_d, grid_size, num_in, bin_counts_d, bin_ptrs_d);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    timer_stop();
  }

  // Launch kernel
  // ----------------------------------------------------------

  timer_start(std::string("Performing ") + mode_info + conf_info + std::string(" computation"));
  switch (mode) {
    case Mode::CPUNormal:
      cpu_normal(in_val_h.data(), in_pos_h.data(), out_h.data(), grid_size, num_in);
      break;
    case Mode::GPUNormal:
      gpu_normal(in_val_d, in_pos_d, out_d, grid_size, num_in);
      break;
    case Mode::GPUCutoff:
      gpu_cutoff(in_val_d, in_pos_d, out_d, grid_size, num_in, cutoff2);
      break;
    case Mode::GPUBinnedCPUPreprocessing:
    case Mode::GPUBinnedGPUPreprocessing:
      gpu_cutoff_binned(bin_ptrs_d, in_val_sorted_d, in_pos_sorted_d, out_d, grid_size, cutoff);
      break;
    default:
      FAIL("Invalid mode " << (int) mode);
  }
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // Copy device variables from host
  // ----------------------------------------

  if (mode != Mode::CPUNormal) {
    THROW_IF_ERROR(cudaMemcpy(out_h.data(), out_d, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    THROW_IF_ERROR(cudaDeviceSynchronize());
  }

  // Verify correctness
  // -----------------------------------------------------

  const auto actual_output = compute_output(in_val_h, in_pos_h, num_in, grid_size);
  verify(actual_output, out_h);

  // Free memory
  // ------------------------------------------------------------

  if (mode != Mode::CPUNormal) {
    if (mode != Mode::GPUBinnedCPUPreprocessing) {
      cudaFree(in_val_d);
      cudaFree(in_pos_d);
    }
    cudaFree(out_d);
    if (mode == Mode::GPUBinnedCPUPreprocessing || mode == Mode::GPUBinnedGPUPreprocessing) {
      cudaFree(in_val_sorted_d);
      cudaFree(in_pos_sorted_d);
      cudaFree(bin_ptrs_d);
      if (mode == Mode::GPUBinnedGPUPreprocessing) {
        cudaFree(bin_counts_d);
      }
    }
  }

  std::cout << "----------------------------------------\n";
  return 0;
}

TEST_CASE("CPUNormal", "[cpu_normal]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::CPUNormal>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::CPUNormal>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::CPUNormal>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::CPUNormal>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::CPUNormal>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::CPUNormal>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::CPUNormal>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::CPUNormal>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::CPUNormal>(696, 1, 232);
  }
}

TEST_CASE("GPUNormal", "[gpu_normal]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUNormal>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUNormal>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUNormal>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUNormal>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUNormal>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUNormal>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUNormal>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUNormal>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUNormal>(696, 1, 232);
  }
}

TEST_CASE("GPUCutoff", "[gpu_cutoff]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUCutoff>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUCutoff>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUCutoff>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUCutoff>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUCutoff>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUCutoff>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUCutoff>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUCutoff>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUCutoff>(696, 1, 232);
  }
}

TEST_CASE("GPUBinnedCPUPreprocessing", "[gpu_binned_cpu_preprocessing]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(696, 1, 232);
  }
}

TEST_CASE("GPUBinnedGPUPreprocessing", "[gpu_binned_gpu_preprocessing]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(696, 1, 232);
  }
}
