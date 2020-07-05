
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

    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor,
                                            1,
                                            1,
                                            1,
                                            1,
                                            1,
                                            1,
                                            CUDNN_CROSS_CORRILATION,
                                            CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t conv_alg;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            filter_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FAST,
                                            0,
                                            &conv_alg));

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