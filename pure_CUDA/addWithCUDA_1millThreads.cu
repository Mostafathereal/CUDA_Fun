#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float*y){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int index = threadIdx.x;
    // int stride = blockDim.x;
    for (int i = index; i < n; i+= stride){
        y[i] = y[i] + x[i];
    }
}

// int validate(float *y){
//     bool flag = true;
//     for(int i = 0; i < 1<<20; i++){
//         flag &= (y[i] == 3.0);
//     }
//     return flag;
// }

int main(void){

    int N = 1<<20;
    
    float *x;
    float *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        add<<<numBlocks, blockSize>>>(N, x, y);

    // wait for GPU to finihs before accessing on host (wait up cpu)
    cudaDeviceSynchronize();

    // std::cout << "\n VALIDATION: " << validate(y) << "\n\n";

    cudaFree(x);
    cudaFree(y);

    

    return 0;



}