#include<iostream>
#include<cmath>
#include<vector>
#include<cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8

__global__ void dTransposeKernel(float* d_in, float* d_out, int n){
    // here index calculation is meant for whole grid (d_in , i.e. n*n size)
    // but each block has only TILE*BLOCK_ROWS threads, but each block is supposed to cater each TILE*TILE (Thread Coarsening) ,
    // i.e. TILE/BLOCK_ROWS memeory transfers needs to be done by each thread, so that Global memory latency can be hidden 

    // so since each thread has to cater TILE/BLOCK_ROWS we will have a for loop for each thread to cater for multiple cells
    
    // usually index calculation is like this , int dirP = blockIdx.P*blockDim.P + threadIdx.P , here P is a random direction 
    // but here for y-dir , we have only BLOCK_ROWS , but each block is supposed to cater a full TILE*TILE, so blockDim x and y needs to be TILE

    int x = blockIdx.x*TILE + threadIdx.x;
    int y = blockIdx.y*TILE + threadIdx.y;

    for(int j = 0 ; j < TILE ; j+= BLOCK_ROWS){
        if(x<n && (y+j)<n){
            d_out[x*n + (y+j)] = d_in[(y+j)*n + x]; // transpose : mat_out [a][b] = mat_in [b][a]
        }
    }

    // done

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
    // N = 8192
    int N = 4096; // size of the matrix for which transpose is being calculated, however this can be replaced with other sizes to compare bandwidth with verion 01
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

    std::cout << "[Version 2: naive transpose] Avg Time: " << avg_msElapsed 
              << " ms | Effective Bandwidth: " << gb_per_sec << " GB/s\n";

    cudaFree(d_in);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}