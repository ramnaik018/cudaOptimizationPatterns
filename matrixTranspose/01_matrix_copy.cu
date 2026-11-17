#include<iostream>
#include<vector>
#include<cmath>
#include<cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8

__global__ void dCopyKernel(const float* d_in, float* d_out, int n){
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    for(int j=0;j<TILE;j+= BLOCK_ROWS){
        if(x<n && (y+j)<n){
            d_out[(y+j)*n + x]=d_in[(y+j)*n +x];
            // n - global width, TILE - block's width
        }
    }
}

void verifyCopy(std::vector<float>& h_in, std::vector<float>& h_out,int n){
    for(int i=0;i<n*n;i++){
        if(std::fabs(h_in[i]-h_out[i]) > 1e-5f) {
            std::cout<<"verification Failed at index "<<i<<std::endl;
            return;
        }
    }
    std::cout<<"Verification SUccessfull"<<std::endl;
    return ;
}

int main(){
    // const int N = 4096;
    const int N = 8192;
    // size of matrix
    const size_t bytes = N * N * sizeof(float);

    // lets define in matrix and out matrix on both host and device
    std::vector<float> h_in(N*N);
    std::vector<float> h_out(N*N, 0.0f); // just initializing with 0, because later, GPUs out matrix will be copied here

    for(int i=0;i<N*N;i++){ h_in[i] = static_cast<float>(i%100); }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    // now copy h_in data to d_in on GPU

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // Now that first part of book keeping is done, lets define blocks and grids

    dim3 block(TILE,BLOCK_ROWS);
    dim3 grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);

    // we will call kernel once now, just to warm up, later we will call it 100 times to actually measure roofline

    dCopyKernel<<<grid,block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop; // cudaEvent_t is like a holder to the actual marker sent into default stream

    cudaEventCreate(&start);
    cudaEventCreate(&stop); // create the actual event objects in GPU , just like cudaMalloc

    int iterations = 100;
    cudaEventRecord(start);
    for(int i=0;i<iterations;i++){
        dCopyKernel<<<grid,block>>> (d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); //CPU wait until stop event is returned

    float msElapsed = 0.0f;
    cudaEventElapsedTime(&msElapsed, start, stop);

    float avg_msElapsed = msElapsed / iterations;

    // calculate Gbs/Sec memory bandwidth (SOL)

    double gb_per_sec = (2.0 * bytes * 1e-9)/(avg_msElapsed * 1e-3);

    std::cout << "[Version 1: Plain Copy] Avg Time: " << avg_msElapsed 
              << " ms | Effective Bandwidth: " << gb_per_sec << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    verifyCopy(h_in, h_out, N);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}