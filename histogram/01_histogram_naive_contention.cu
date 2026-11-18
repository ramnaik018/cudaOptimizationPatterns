#include<iostream>
#include<vector>
#include<cuda_runtime.h>

#define BIN 256
#define BLOCK 256


__global__ void dHistKernel(unsigned char* d_in, unsigned* d_hist, unsigned n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x;

    for(int i=id ;i<n; i+= stride){
        atomicAdd(&d_hist[d_in[i]],1u);
    }
}

void verifyHistograms(std::vector<unsigned char> &h_in, std::vector<unsigned> &h_hist){
    std::vector<unsigned> ref(BIN, 0);
    for (unsigned char val : h_in) {
        ref[val]++;
    }
    for (int i = 0; i < BIN; ++i) {
        if (ref[i] != h_hist[i]) {
            std::cerr << "Verification FAILED at bin " << i
                      << " (expected " << ref[i] << ", got " << h_hist[i] << ")\n";
            return;
        }
    }
    std::cout << "Verification PASSED!\n";

}

int main(){
    const int N = (1<<26); // which equals 2^26 = 2^20 * 2^6 = 10^6 * 64 = 64Million 
    const size_t in_bytes = N * sizeof(unsigned char);
    const size_t hist_bytes = BIN * sizeof(unsigned);

    // now intialise host vectors for input array and histogram 

    std::vector<unsigned char> h_in(N);
    std::vector<unsigned> h_hist (BIN, 0);

    // now lets populate the input vector , 90% skewed , and 10% filled with balenced values from 0 ... 255

    for(int i =0;i<N;i++){
        h_in[i] = (i%10 == 0)?(i%BIN):0; // 0 gets implicitly casted by c++ , because 0 is a constexpr -> compile time constant
    }

    // well host side data is populated, now its time to populate device side data

    unsigned* d_hist;
    unsigned char* d_in;

    cudaMalloc(&d_hist, hist_bytes);
    cudaMalloc(&d_in, in_bytes);

    // we need to also make sure device histogram is set by 0 initially

    cudaMemset(d_hist , 0, hist_bytes);

    // to utilise grid strided loop, we will intialise grid size 
    int gridSize = 256;

    // lets transfer the in data to device ,
    cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);

    // lets warp up the device using a single call to kernel to populate d_hist using d_in
    dHistKernel<<<gridSize,BLOCK>>>(d_in, d_hist, N);
    cudaDeviceSynchronize();

    // lets measure the bandwidth

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned iterations = 100;
    // since for each ietration we need to set back the d_hist to 0's, we cannot place markers of event out of the loop

    float totalMs = 0.0f;
    for(unsigned i=0;i<iterations;i++){
        float currentMs = 0.0f;
        cudaMemset(d_hist, 0 , hist_bytes);
        cudaEventRecord(start);
        dHistKernel<<<gridSize,BLOCK>>>(d_in, d_hist, N);
        cudaEventRecord(stop); // we dont need cudaDeviceSynchronize() here, coz cudaEventRecord(stop) already tracks streams completion

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&currentMs, start, stop);

        totalMs+= currentMs;
    }

    cudaMemcpy(h_hist.data() , d_hist,hist_bytes, cudaMemcpyDeviceToHost);

    verifyHistograms(h_in, h_hist);

    float avg_ms =  totalMs / iterations;

    std::cout << "Average time taken by each kernel (i.e. histogram creation is) : "<<avg_ms<<" ms"<<std::endl;

    float gb_per_sec =  ((in_bytes+hist_bytes) * 1e-9) / (avg_ms * 1e-3 ) ; 

    std::cout << "[naive Histogram bandwidth ] is : " << gb_per_sec << "GB per Sec" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_hist);

    return 0;
}