
#include <cudnn.h>
#include <iostream>
#include <opencv2/opencv.hpp>
 
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat ldimag(const char* image_path){
    cv::Mat image = cv::imread(image_path);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

int main(void){
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    cv::Mat image = ldimag("conure.jpg");

    // define descriptors for input/output tensors, and for filter tensors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                            CUDNN_TENSOR_NCHW, 
                                            CUDNN_DATA_FLOAT, 
                                            1, 
                                            3,
                                            image.rows,
                                            image.cols));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, 
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            1, 
                                            3, 
                                            image.rows,
                                            image.cols));

    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW,
                                            3,
                                            3,
                                            3,
                                            3));
    
    // define type of convolution (same padding, stride of 1 (for h and w), and no dilation (=1))                                         
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor))
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                            1,
                                            1,
                                            1,
                                            1,
                                            1,
                                            1,
                                            CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT));

    // define the convolution algorithm, using the type of conv described above                                        
    cudnnConvolutionFwdAlgo_t conv_alg;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            filter_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            0,
                                            &conv_alg));
    // allocate buffer memeory for system to execute algorithm, first find size of workspace required (depends on conv_alg)
    size_t ws_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_descriptor, 
                                            filter_descriptor, 
                                            conv_descriptor,
                                            output_descriptor,
                                            conv_alg,
                                            &ws_bytes));
                                        
    std::cout << "WS size: " << (ws_bytes / (1<<20)) << "- MB" << std::endl;

    // size of input img in bytes
    imsize = image.rows * image.cols * 3 * sizeof(float)

    void* d_ws = nullptr;
    cudaMalloc(&d_ws, ws_bytes);

    // allocate h * w * channels * float size to the input tensor
    float* d_input = nullptr;
    cudaMalloc(&d_input, imsize);

    // since it is a `same` convolution, use same amount of memory for output 
    float* d_output = nullptr;
    cudaMalloc(&d_input, imsize);

    cudaMemcpy(d_input, image.ptr<float>0, imsize, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, imsize);

    // 3x3 size of kernel, x3 to match #channels of input image, (resulting in a 1 channel out-img)
    // again x3 to result in the same number of channels in output as input
    float kernel_temp[3][3][[3][3] = 
    {{{{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}},     
     {{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}},     
     {{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}}},     
     {{{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}},     
     {{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}},     
     {{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}}},     
     {{{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}},     
     {{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}},     
     {{-2, 0, 2},
     {-5, 0, 5},
     {-2, 0, 2}}}}

    float* d_kernel = nullptr;
    cudaMalloc(d_kernel, sizeof(kernel_temp))
    cudaMemcpy(d_kernel, kernel_temp, sizeof(kernel_temp), cudaMemcpyHostToDevice)

    //performing the convolution
    float alpha, beta = 1, 0;
    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_kernel, 
                                        conv_descriptor,
                                        conv_alg,
                                        d_ws,
                                        &beta,
                                        output_descriptor,
                                        d_output));


    // float* h_output = new float[imsize];
    // cudaMemcpy()
    


    



    return 0;
}