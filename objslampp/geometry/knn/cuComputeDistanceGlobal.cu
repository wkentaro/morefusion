// #define IDX2D(i, j, dj) (dj * i + j)
// #define IDX3D(i, j, k, dj, dk) (IDX2D(IDX2D(i, j, dj), k, dk))

// #define BLOCK 512
// #define MAX_STREAMS 512
#define BLOCK_DIM 16


/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
extern "C" __global__
void cuComputeDistanceGlobal( float* A, int wA,
    float* B, int wB, int dim, float* AB){

// Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
__shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
__shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];


  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  float tmp;
  float ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A  = BLOCK_DIM * wA;
  step_B  = BLOCK_DIM * wB;
  end_A   = begin_A + (dim-1) * wA;

    // Conditions
  int cond0 = (begin_A + tx < wA); // used to write in shared memory
  int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
  int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix

  // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
    if (a/wA + ty < dim){
      shared_A[ty][tx] = (cond0)? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1)? B[b + wB * ty + tx] : 0;
    }
    else{
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
    if (cond2 && cond1){
      for (int k = 0; k < BLOCK_DIM; ++k){
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp*tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1)
    AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
}
