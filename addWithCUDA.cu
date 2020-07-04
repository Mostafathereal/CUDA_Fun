#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float*y){
    for (int i = 0; i < n; i++){
        y[i] = y[i] + x[i];
    }
}

int main(void){

    int N = 1<<20;
    
    float *x = new float[N];
    float *y = new float[N];

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<32, 1>>>(N, x, y);

    // wait for GPU to finihs before accessing on host (wait up cpu)

    cudaFree(x);
    cudaFree(y);

    return 0;



}