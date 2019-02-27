#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the global queue
  unsigned int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (idx == 0)
    *numNextLevelNodes = 0;
  __syncthreads();
  while(idx < *numCurrLevelNodes){
    const unsigned int currentNode = currLevelNodes[idx];
    for (unsigned int i = nodePtrs[currentNode];i < nodePtrs[currentNode + 1];i++){
      const unsigned int currentNeighbor = nodeNeighbors[i];
      if(!atomicExch(&(nodeVisited[currentNeighbor]), 1)){
        const unsigned int tail = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[tail] = currentNeighbor;
      }
    }
    idx += gridDim.x * blockDim.x;
  }
}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the block queue
  // If full, add it to the global queue

  // Calculate space for block queue to go into global queue

  // Store block queue in global queue
  __shared__ unsigned int blockQueue[BQ_CAPACITY];
  __shared__ unsigned int blockQueueTail, globalQueueTail;
  
  // Initialization
  unsigned int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if(threadIdx.x == 0){
    blockQueueTail = 0;
    if(idx == 0)
      *numNextLevelNodes = 0;
  }
  __syncthreads();

  // Loop
  while(idx < *numCurrLevelNodes){
    const unsigned int currentNode = currLevelNodes[idx];
    for(unsigned int i = nodePtrs[currentNode];i < nodePtrs[currentNode + 1];i++){
      const unsigned int currentNeighbor = nodeNeighbors[i];
      if(!atomicExch(&(nodeVisited[currentNeighbor]), 1)){
          const unsigned int bTail = atomicAdd(&(blockQueueTail), 1);
        if(bTail < BQ_CAPACITY){
          blockQueue[bTail] = currentNeighbor;
        }
        else{
          blockQueueTail = BQ_CAPACITY;
          nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = currentNeighbor;
        }
      }
    }
    idx += blockDim.x * gridDim.x;
  }
  __syncthreads();
  
  if(threadIdx.x == 0)
    globalQueueTail = atomicAdd(numNextLevelNodes, blockQueueTail);
  __syncthreads();
  
  for(unsigned int i = threadIdx.x;i < blockQueueTail;i += blockDim.x)
    nextLevelNodes[globalQueueTail + i] = blockQueue[i];
    
  
}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses one queue per warp

  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the warp queue
  // If full, add it to the block queue
  // If full, add it to the global queue

  // Calculate space for warp queue to go into block queue

  // Store warp queue in block queue
  // If full, add it to the global queue

  // Calculate space for block queue to go into global queue
  // Saturate block queue counter
  // Calculate space for global queue

  // Store block queue in global queue
  __shared__ unsigned int warpQueue[WQ_CAPACITY][NUM_WARPS];
  __shared__ unsigned int warpQueueTail[NUM_WARPS];
  __shared__ unsigned int warpToBlockQueueTail[NUM_WARPS];
  __shared__ unsigned int blockQueue[BQ_CAPACITY];
  __shared__ unsigned int blockQueueTail, blockToGlobalQueueTail;

  unsigned int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  const unsigned int warpQueueIdx = threadIdx.x % NUM_WARPS;

  if(threadIdx.x / NUM_WARPS == 0){
    warpQueueTail[threadIdx.x] = 0;
    if(threadIdx.x == 0){
      blockQueueTail = 0;
      if(idx == 0)
        *numNextLevelNodes = 0;
    }
  }
  __syncthreads();

  for(;idx < *numCurrLevelNodes;idx += blockDim.x * gridDim.x){
    const unsigned int currentNode = currLevelNodes[idx];
    for(unsigned int i = nodePtrs[currentNode];i < nodePtrs[currentNode + 1];i++){
      const unsigned int currentNeighbor = nodeNeighbors[i];
      if(!atomicExch(&(nodeVisited[currentNeighbor]), 1)){
        const unsigned int wTail = atomicAdd(&(warpQueueTail[warpQueueIdx]), 1);
        if(wTail < WQ_CAPACITY){
          warpQueue[wTail][warpQueueIdx] = currentNeighbor;
        }
        else{
          warpQueueTail[warpQueueIdx] = WQ_CAPACITY;
          const unsigned int bTail = atomicAdd(&(blockQueueTail), 1);
          if(bTail < BQ_CAPACITY){
            blockQueue[bTail] = currentNeighbor;
          }
          else{
            blockQueueTail = BQ_CAPACITY;
            nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = currentNeighbor;
          }
        }
      }
    }
  }
  __syncthreads();

  // Store warp queue into block queue
  // First, calculate tail index by exclusive scan
  // WARINNING: You cannot use Kogge-Stone to do the scan because you cannot sychronize part of the (NUM_WARPS for here) threads
  if(threadIdx.x == 0){
    warpToBlockQueueTail[0] = blockQueueTail;
    for(unsigned int i = 1;i < NUM_WARPS;i++)
      warpToBlockQueueTail[i] = warpToBlockQueueTail[i - 1] + warpQueueTail[i - 1];
  }
  __syncthreads();

  if(threadIdx.x == 0){
    if(warpToBlockQueueTail[NUM_WARPS - 1] + warpQueueTail[NUM_WARPS - 1] < BQ_CAPACITY)
      blockQueueTail = warpToBlockQueueTail[NUM_WARPS - 1] + warpQueueTail[NUM_WARPS - 1];
    else
      blockQueueTail = BQ_CAPACITY; 
    blockToGlobalQueueTail = atomicAdd(numNextLevelNodes,  blockQueueTail);
  }
  const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  for(unsigned int i = threadIdx.x % WARP_SIZE;i < warpQueueTail[warpIdx];i += WARP_SIZE){
    const unsigned int warpToBlockQueueIdx = warpToBlockQueueTail[warpIdx] + i;
    if(warpToBlockQueueIdx < BQ_CAPACITY)
      blockQueue[warpToBlockQueueIdx] = warpQueue[i][warpIdx];
    else
      nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = warpQueue[i][warpIdx];
  }
  __syncthreads();

  for(unsigned int i = threadIdx.x;i < blockQueueTail;i += blockDim.x)
    nextLevelNodes[blockToGlobalQueueTail + i] = blockQueue[i];
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
