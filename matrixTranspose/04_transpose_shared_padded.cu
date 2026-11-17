#include<iostream>
#include<cmath>
#include<vector>
#include<cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8

__global__ void dTransposeKernel(float* d_in, float* d_out, int n){

    __shared__ float tile[TILE][TILE+1]; // THIS IS THE ONLY OPTIMIZATION 

   int x = blockIdx.x*TILE + threadIdx.x;
   int y = blockIdx.y*TILE + threadIdx.y;

   for(int j=0;j<TILE;j+=BLOCK_ROWS){
    if(x<n && (y+j)<n) tile[threadIdx.y+j][threadIdx.x] = d_in[(y+j)*n + x]; 
    // till here data of a tile is copied to shared memory as is (no transpose yet)
   }
   __syncthreads();

   x = blockIdx.y*TILE + threadIdx.x;
   y = blockIdx.x*TILE + threadIdx.y; // swapping of block is done , now while we assign actual
   // values to d_out, we will do intra-tile transpose 

   for(int j=0;j<TILE;j+=BLOCK_ROWS){
    if(x<n && (y+j)<n) d_out[(y+j)*n + x] = tile[threadIdx.x][threadIdx.y+j];
   }

   // DONE, we have transposed one single tile per thread block. we used thread coarsing to deal
   // with actual matrix size even if we have less number of threads per cell in a tile

}

void verifyTranspose(std::vector<float>& h_in, std::vector<float>& h_out, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(std::fabs(h_in[i*n + j] - h_out[j*n + i])>1e-6) { // sabs takes only one argument 
                std::cout<< "Transpose failed for "<< h_in[i*n + j]<<" and "<<h_out[j*n + i]<<std::endl;
                return;
            }
        }
    }
    std::cout<<"Transpose successful..!"<<std::endl;
    return;
}

int main(){
    int N = 8192; // size of the matrix for which transpose is being calculated, however this can be replaced with other sizes to compare bandwidth with verion 01
    //int N = 4096;
    // now lets define host and device (in and out) matrices

    std::vector<float> h_in (N*N);
    std::vector<float> h_out (N*N , 0.0f);

    // now lets fill h_in with some values maybe i%100

    for(int i=0;i<N*N;i++){ h_in[i] = static_cast<float>(i%100); } // need to convert int into float
    // now lets define device in and out vectors with sizes

    size_t bytes = static_cast<size_t> (N)*N*sizeof(float);

    float *d_in, *d_out;
    // now lts allocate some memory space for in and out and copy the h_in into d_in

    cudaMalloc(&d_in , bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // as of this point, major bookkeeping is done, devices mats are ready to be transposed lets do on warm up for GPU for one time, then run same kernel
    // for 100 ietrations to measure accurate Mem bandwidth and compare with our SOL from Version 01

    dim3 block(TILE,BLOCK_ROWS);
    dim3 grid((N+TILE-1)/TILE , (N+TILE-1)/TILE);

    dTransposeKernel<<<grid , block>>>(d_in , d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    verifyTranspose(h_in,h_out,N);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;

    cudaEventRecord(start);
    for(int i=0;i<iterations;i++){
        dTransposeKernel<<<grid,block>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float msElapsed = 0.0f;
    cudaEventElapsedTime(&msElapsed,start,stop);
    float avg_msElapsed = msElapsed/iterations;

    float gb_per_sec = (2.0f * bytes * 1e-9) / (avg_msElapsed * 1e-3) ;

    std::cout << "[Version 4: padded shared memory transpose] Avg Time: " << avg_msElapsed 
              << " ms | Effective Bandwidth: " << gb_per_sec << " GB/s\n";

    cudaFree(d_in);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}