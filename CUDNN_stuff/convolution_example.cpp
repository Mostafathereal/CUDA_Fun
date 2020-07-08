
#include <cudnn.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
 
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void save_image(const char* output_filename,
    float* buffer,
    int height,
    int width) {
cv::Mat output_image(height, width, CV_32FC3, buffer);
// Make negative values zero.
cv::threshold(output_image,
    output_image,
    /*threshold=*/0,
    /*maxval=*/0,
    cv::THRESH_TOZERO);
cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
output_image.convertTo(output_image, CV_8UC3);
cv::imwrite(output_filename, output_image);
std::cerr << "Wrote output to " << output_filename << std::endl;
}

cv::Mat ldimag(const char* image_path){
    cv::Mat image = cv::imread(image_path);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

int main(void){
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cv::Mat image = ldimag("conure.jpg");

    // define descriptors for input/output tensors, and for filter tensors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                            CUDNN_TENSOR_NHWC, 
                                            CUDNN_DATA_FLOAT, 
                                            1, 
                                            3,
                                            image.rows,
                                            image.cols));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, 
                                            CUDNN_TENSOR_NHWC,
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
    int imsize = image.rows * image.cols * 3 * sizeof(float);

    void* d_ws = nullptr;
    cudaMalloc(&d_ws, ws_bytes);

    // allocate h * w * channels * float size to the input tensor
    float* d_input = nullptr;
    cudaMalloc(&d_input, imsize);

    // since it is a `same` convolution, use same amount of memory for output 
    float* d_output = nullptr;
    cudaMalloc(&d_output, imsize);

    float* d_output2 = nullptr;
    cudaMalloc(&d_output2, imsize);

    float* d_output3 = nullptr;
    cudaMalloc(&d_output3, imsize);

    float* d_output4 = nullptr;
    cudaMalloc(&d_output4, imsize);

    cudaMemcpy(d_input, image.ptr<float>(0), imsize, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, imsize);

    cudaMemset(d_output2, 0, imsize);

    // 3x3 size of kernel, x3 to match #channels of input imafloat* h_output = new float[imsize];ge, (resulting in a 1 channel out-img)
    // again x3 to result in the same number of channels in output as input
float kernel_temp[3][3][3][3] = { {{{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}, 
                                  {{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}, 
                                  {{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}}, 
                                  {{{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}, 
                                  {{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}, 
                                  {{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}}, 
                                  {{{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}, 
                                  {{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}, 
                                  {{-2, 0, 2}, {-5, 0, 5}, {-2, 0, 2}}}}; 

float kernel_temp2[3][3][3][3] = { {{{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}, 
                                  {{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}, 
                                  {{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}}, 
                                  {{{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}, 
                                  {{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}, 
                                  {{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}}, 
                                  {{{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}, 
                                  {{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}, 
                                  {{-2, -5, -2}, {0, 0, 0}, {2, 5, 2}}}};

float kernel_temp3[3][3][3][3] = { {{{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}, 
                                  {{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}, 
                                  {{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}}, 
                                  {{{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}, 
                                  {{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}, 
                                  {{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}}, 
                                  {{{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}, 
                                  {{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}, 
                                  {{2, 5, 2}, {0, 0, 0}, {-2, -5, -2}}}}; 

float kernel_temp4[3][3][3][3] = { {{{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}, 
                                  {{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}, 
                                  {{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}}, 
                                  {{{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}, 
                                  {{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}, 
                                  {{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}}, 
                                  {{{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}, 
                                  {{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}, 
                                  {{2, 0, -2}, {5, 0, -5}, {2, 0, -2}}}}; 

    float* d_kernel = nullptr;
    cudaMalloc(&d_kernel, sizeof(kernel_temp));
    cudaMemcpy(d_kernel, kernel_temp, sizeof(kernel_temp), cudaMemcpyHostToDevice);

    float* d_kernel2 = nullptr;
    cudaMalloc(&d_kernel2, sizeof(kernel_temp2));
    cudaMemcpy(d_kernel2, kernel_temp2, sizeof(kernel_temp2), cudaMemcpyHostToDevice);

    float* d_kernel3 = nullptr;
    cudaMalloc(&d_kernel3, sizeof(kernel_temp3));
    cudaMemcpy(d_kernel3, kernel_temp3, sizeof(kernel_temp3), cudaMemcpyHostToDevice);

    float* d_kernel4 = nullptr;
    cudaMalloc(&d_kernel4, sizeof(kernel_temp4));
    cudaMemcpy(d_kernel4, kernel_temp4, sizeof(kernel_temp4), cudaMemcpyHostToDevice);

    //performing the convolution
    float alpha, beta; 
    alpha = 1;
    beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_kernel, 
                                        conv_descriptor,
                                        conv_alg,
                                        d_ws,
                                        ws_bytes,
                                        &beta,
                                        output_descriptor,
                                        d_output));

    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_kernel2, 
                                        conv_descriptor,
                                        conv_alg,
                                        d_ws,
                                        ws_bytes,
                                        &beta,
                                        output_descriptor,
                                        d_output2));

    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_kernel4, 
                                        conv_descriptor,
                                        conv_alg,
                                        d_ws,
                                        ws_bytes,
                                        &beta,
                                        output_descriptor,
                                        d_output4));


    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_kernel3, 
                                        conv_descriptor,
                                        conv_alg,
                                        d_ws,
                                        ws_bytes,
                                        &beta,
                                        output_descriptor,
                                        d_output3));

    std::cout << "\n\n hehe after conv \n\n\n";


    float* h_output = new float[imsize];
    float* h_output2 = new float[imsize];
    float* h_output3 = new float[imsize];
    float* h_output4 = new float[imsize];

    std::cout << "\n\n hehe after conv 1\n\n\n";

    cudaMemcpy(h_output, d_output, imsize, cudaMemcpyDeviceToHost);

    std::cout << "\n\n hehe after conv 2\n\n\n";

    save_image("conure(out).png", h_output, image.rows, image.cols);

    cudaMemcpy(h_output2, d_output2, imsize, cudaMemcpyDeviceToHost);
    save_image("conure(out)2.png", h_output2, image.rows, image.cols);

    cudaMemcpy(h_output3, d_output3, imsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output4, d_output4, imsize, cudaMemcpyDeviceToHost);

    save_image("conure(out)3.png", h_output3, image.rows, image.cols);
    save_image("conure(out)4.png", h_output4, image.rows, image.cols);

    for (int count = 0; count < imsize; count++){
      h_output2[count] = std::max(std::max(std::max(h_output[count], h_output2[count]), h_output3[count]), h_output4[count]);

    }

    save_image("conure(out)combined.png", h_output2, image.rows, image.cols);

    std::cout << "\n\n hehe after conv 3\n\n\n" << imsize << "\n\n";


    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_ws);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);

    cudnnDestroy(cudnn);

    return 0;
}