
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

    

    

    



    return 0;
}