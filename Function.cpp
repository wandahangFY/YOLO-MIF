#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

__global__
void blurKernel(float* img, int width, int height, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < height - 1 && j >= 1 && j < width - 1) {
        float sum = 0;
        for (int k = -1; k <= 1; ++k) {
            for (int l = -1; l <= 1; ++l) {
                sum += img[(i + k) * width + (j + l)];
            }
        }
        sum /= 9;
        result[i * width + j] = sum;
    }
}

__global__
void receptiveFieldKernel(float* img, int width, int height, float* kernel, int R, int r, float fac_r, float fac_R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= R && i < height - R && j >= R && j < width - R) {
        float sum = 0;
        for (int k = -R; k <= R; ++k) {
            for (int l = -R; l <= R; ++l) {
                float dis = sqrt(pow(k, 2) + pow(l, 2));
                if (dis <= r) {
                    sum += img[(i + k) * width + (j + l)] * fac_r;
                } else if (dis > r && dis <= R) {
                    sum += img[(i + k) * width + (j + l)] * fac_R;
                }
            }
        }
        sum /= (2 * R + 1) * (2 * R + 1);
        kernel[i * width + j] = sum;
    }
}

Mat blurUsingCUDA(Mat img) {
    Mat result(img.size(), CV_32F);

    float* d_img, *d_result;
    cudaMalloc(&d_img, img.rows * img.cols * sizeof(float));
    cudaMalloc(&d_result, img.rows * img.cols * sizeof(float));

    cudaMemcpy(d_img, img.data, img.rows * img.cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
    blurKernel<<<gridSize, blockSize>>>(d_img, img.cols, img.rows, d_result);

    cudaMemcpy(result.data, d_result, img.rows * img.cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_result);

    return result;
}

Mat receptiveFieldUsingCUDA(Mat img, int R=3, int r=1, float fac_r=-1, float fac_R=6) {
    Mat kernel(img.size(), CV_32F);

    float* d_img, *d_kernel;
    cudaMalloc(&d_img, img.rows * img.cols * sizeof(float));
    cudaMalloc(&d_kernel, img.rows * img.cols * sizeof(float));

    cudaMemcpy(d_img, img.data, img.rows * img.cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
    receptiveFieldKernel<<<gridSize, blockSize>>>(d_img, img.cols, img.rows, d_kernel, R, r, fac_r, fac_R);

    cudaMemcpy(kernel.data, d_kernel, img.rows * img.cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_kernel);

    return kernel;
}

int main() {
    Mat img = imread("input.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Failed to read image." << endl;
        return -1;
    }

    Mat blurred = blurUsingCUDA(img);

    Mat rec = receptiveFieldUsingCUDA(blurred);

    Mat result;
    vector<Mat> channels = {img, blurred, rec};
    merge(channels, result);

    imwrite("output.png", result);

    return 0;
}




{
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// CPU版本的receptiveField函数
Mat receptiveField(Mat img, int R=3, int r=1, float fac_r=-1, float fac_R=6) {
    Mat kernel = Mat::zeros(2 * R + 1, 2 * R + 1, CV_32F);  // 定义卷积核矩阵
    float sum = 0;  // 卷积核元素之和

    // 计算网格坐标以及对应的距离矩阵
    for (int i = 0; i < 2 * R + 1; ++i) {
        for (int j = 0; j < 2 * R + 1; ++j) {
            float dis = sqrt(pow(i - (R + 1), 2) + pow(j - (R + 1), 2));  // 计算距离
            if (dis <= r) {
                kernel.at<float>(i, j) = fac_r;  // 设置卷积核元素值
            } else if (dis > r && dis <= R) {
                kernel.at<float>(i, j) = fac_R;  // 设置卷积核元素值
            }
            sum += kernel.at<float>(i, j);  // 累加卷积核元素
        }
    }
    kernel /= sum;  // 归一化卷积核

    Mat out;
    filter2D(img, out, -1, kernel);  // 进行卷积操作
    return out;
}

int main() {
    Mat img = imread("input.png", IMREAD_GRAYSCALE);  // 读取单通道灰度图像
    if (img.empty()) {
        cout << "Failed to read image." << endl;
        return -1;
    }

    Mat blur;
    cv::blur(img, blur, Size(3, 3));  // 进行模糊处理

    Mat rec = receptiveField(img);  // 进行细节增强

    Mat result;
    vector<Mat> channels = {img, blur, rec};
    merge(channels, result);  // 合并三个通道

    imwrite("output.png", result);  // 输出结果图像

    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

__global__
void receptiveFieldKernel(float* img, int width, int height, float* kernel, int R, int r, float fac_r, float fac_R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= R && i < height - R && j >= R && j < width - R) {
        float sum = 0;
        for (int k = -R; k <= R; ++k) {
            for (int l = -R; l <= R; ++l) {
                float dis = sqrt(pow(k, 2) + pow(l, 2));
                if (dis <= r) {
                    sum += img[(i + k) * width + (j + l)] * fac_r;
                } else if (dis > r && dis <= R) {
                    sum += img[(i + k) * width + (j + l)] * fac_R;
                }
            }
        }
        sum /= (2 * R + 1) * (2 * R + 1);
        kernel[i * width + j] = sum;
    }
}

// CUDA版本的receptiveField函数
Mat receptiveField(Mat img, int R=3, int r=1, float fac_r=-1, float fac_R=6) {
    Mat kernel = Mat::zeros(img.size(), CV_32F);  // 定义卷积核矩阵

    float* d_img, *d_kernel;
    cudaMalloc(&d_img, img.rows * img.cols * sizeof(float));  // 分配设备内存
    cudaMalloc(&d_kernel, img.rows * img.cols * sizeof(float));

    cudaMemcpy(d_img, img.data, img.rows * img.cols * sizeof(float), cudaMemcpyHostToDevice);  // 将数据从主机内存复制到设备内存

    dim3 blockSize(32, 32);
    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
    receptiveFieldKernel<<<gridSize, blockSize>>>(d_img, img.cols, img.rows, d_kernel, R, r, fac_r, fac_R);  // 调用核函数进行卷积操作

    cudaMemcpy(kernel.data, d_kernel, img.rows * img.cols * sizeof(float), cudaMemcpyDeviceToHost);  // 将数据从设备内存复制到主机内存

    cudaFree(d_img);  // 释放设备内存
    cudaFree(d_kernel);

    return kernel;
}

int main() {
    Mat img = imread("input.png", IMREAD_GRAYSCALE);  // 读取单通道灰度图像
    if (img.empty()) {
        cout << "Failed to read image." << endl;
        return -1;
    }

    Mat blur;
    cv::blur(img, blur, Size(3, 3));  // 进行模糊处理

    Mat rec = receptiveField(img);  // 进行细节增强

    Mat result;
    vector<Mat> channels = {img, blur, rec};
    merge(channels, result);  // 合并三个通道

    imwrite("output.png", result);  // 输出结果图像

    return 0;
}



}