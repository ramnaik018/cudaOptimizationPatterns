#include<iostream>
#include<cmath>
#include<vector>
#include<cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void gemmKernel(float* a, float* b, float* c, int M , int K, int N){
    // lets delieratly assign strided access based indexes to x and y,
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y; 

    
    if(row<M && col<N){
        float sum = 0.0f;
        for(int k = 0 ;k<K; k++){
            sum += a[row*K + k] * b[k*N + col]; // a[row][k] * b[k][col] ; a width is K and b width is N
        }
        c[row*N + col] = sum;
    }
}
    

void verifySGEMM(const std::vector<float>& A, const std::vector<float>& B,
                 const std::vector<float>& C, int M, int N, int K) {
    // Spot-check random/sample indices so CPU verification runs in milliseconds
    for (int idx = 0; idx < 100; ++idx) {
        int r = (idx * 37) % M;
        int c = (idx * 73) % N;
        float expected = 0.0f;
        for (int k = 0; k < K; ++k) {
            expected += A[r * K + k] * B[k * N + c];
        }
        if (std::fabs(expected - C[r * N + c]) > 1e-2f) {
            std::cerr << "Verification FAILED at (" << r << ", " << c
                      << ") expected " << expected << ", got " << C[r * N + c] << "\n";
            return;
        }
    }
    std::cout << "Verification PASSED!\n";
}


int main(){
    const int M = 1024 , K = 1024 , N = 1024;
    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    std::vector<float> h_a(M*K), h_b(K*N), h_c(M*N , 0.0f);

    for(int i = 0;i<M*K;i++) {h_a[i] = static_cast<float>((i%10) - 5) * 0.1f; }
    for(int i = 0;i<K*N;i++) {h_b[i] = static_cast<float>((i%7) - 3) * 0.1f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy( d_a, h_a.data(), bytes_a , cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b.data(), bytes_b , cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE);

    // now warm up kernel launch 

    gemmKernel<<<grid,block>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();

    //Benchmarking and measuring performance FLOPS / second

    int iterations = 20;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i =0;i<iterations; i++){
        gemmKernel<<<grid,block>>>(d_a, d_b, d_c, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msElapsed = 0.0f;

    cudaEventElapsedTime(&msElapsed, start, stop);

    float avgMs = msElapsed / iterations;

   // float gb_per_sec = bytes_a + bytes_b + bytes_c; dont know yet how to claculate

    double flops = 2.0 * M * N * K;
    double seconds = avgMs * 1e-3;
    double GflopPerSec = (flops * 1e-9) / seconds;
   
   std::cout<< "performanc : "<< GflopPerSec<<" GBytesPerSec"<< std::endl;

   cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

   verifySGEMM(h_a, h_b, h_c , M, N , K);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   return 0;

}