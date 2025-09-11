## 쿠다 코어

**목표:** 기존 CPU 기반 라이브러리 그대로 두고, 각 함수 내부 연산만 GPU 병렬화(CUDA)

### 카테고리별 전략

1. **픽셀 단순 연산(Pixel-wise)**
    - 예: Binarization, Equalization, Histogram
    - 전략: GPU 쓰레드 1개 = 픽셀 1개 처리
    - GPU 메모리에 이미지 올리고 커널 호출 → 결과 복사 back
2. **필터/컨볼루션(Filter)**
    - 예: Sobel, Laplacian, Gaussian Blur
    - 전략: 각 스레드가 픽셀 1개 처리, 커널 영역 합산
    - 최적화: shared memory로 주변 픽셀 접근 최소화
3. **구조/변환(Structural/Transform)**
    - 예: Dilation, Erosion, NCC, SAD, SSD
    - 전략: 픽셀 영역 단위 반복 → 블록 단위 스레드 병렬화
    - 최적화: 필요한 경우 shared memory, block synchronization 활용

**공통 전략**

- 기존 API 그대로 유지
- CPU → GPU 전환만 내부에서 처리
- CUDA 라이브러리 가능하면 cuFFT/cuBLAS/cuDNN 활용
- GPU 메모리 할당/해제 주의

---

## 2️⃣ Gemini에게 줄 프롬프트

```
안녕 Gemini. 내가 만든 이미지 처리 라이브러리가 있어.
라이브러리 구조는 CPU 기반으로 되어 있고, ApplyBinarization, ApplyEqualization, ApplyHistogram 등
픽셀 단순 연산, Sobel, Laplacian, Gaussian Blur 같은 필터, Dilation, Erosion, NCC/SAD/SSD 같은
구조/변환 연산이 포함되어 있어.

내 목표는 기존 함수 이름과 API는 그대로 두고,
함수 내부에서 GPU(CUDA)로 병렬 연산하도록 변경하는 거야.

각 카테고리별 CUDA 최적화 전략과 예시 코드를 보여주고,
픽셀 단순 연산, 필터, 구조/변환 각각 GPU로 옮길 때 주의할 점과
성능 최적화 방법까지 설명해줄래?

또, GPU 메모리 관리, shared memory 활용, 스레드 블록 구성에 대한 구체적 예시도 포함해줘.

```

## 기본 뼈대

```cpp
// GPU_Kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// ---------------- Pixel-wise Operations ----------------
__global__ void BinarizationKernel(unsigned char* data, int width, int height, int stride, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * stride + x;
    data[idx] = (data[idx] > threshold) ? 255 : 0;
}

__global__ void EqualizationKernel(unsigned char* data, int width, int height, int stride, int* cdf, int cdfMin, int totalPixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * stride + x;
    int val = data[idx];
    int newVal = round(((double)cdf[val] - cdfMin) / (totalPixels - cdfMin) * 255.0);
    if (newVal < 0) newVal = 0;
    if (newVal > 255) newVal = 255;
    data[idx] = static_cast<unsigned char>(newVal);
}

// ---------------- Filter / Convolution ----------------
__global__ void ConvolutionKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride,
                                  const double* kernel, int kernelSize, double kernelSum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int center = kernelSize / 2;

    if (x < center || y < center || x >= width - center || y >= height - center) return;

    double sum = 0.0;
    for (int ky = -center; ky <= center; ++ky) {
        for (int kx = -center; kx <= center; ++kx) {
            int idx = (y + ky) * stride + (x + kx);
            int kIdx = (ky + center) * kernelSize + (kx + center);
            sum += kernel[kIdx] * input[idx];
        }
    }

    if (kernelSum != 0) sum /= kernelSum;
    if (sum > 255) sum = 255;
    if (sum < 0) sum = 0;

    output[y * stride + x] = static_cast<unsigned char>(sum);
}

// ---------------- Structural / Transform ----------------
__global__ void DilationKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int center = kernelSize / 2;

    if (x < center || y < center || x >= width - center || y >= height - center) return;

    unsigned char maxVal = 0;
    for (int ky = -center; ky <= center; ++ky) {
        for (int kx = -center; kx <= center; ++kx) {
            int idx = (y + ky) * stride + (x + kx);
            if (input[idx] > maxVal) maxVal = input[idx];
        }
    }
    output[y * stride + x] = maxVal;
}

__global__ void ErosionKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int center = kernelSize / 2;

    if (x < center || y < center || x >= width - center || y >= height - center) return;

    unsigned char minVal = 255;
    for (int ky = -center; ky <= center; ++ky) {
        for (int kx = -center; kx <= center; ++kx) {
            int idx = (y + ky) * stride + (x + kx);
            if (input[idx] < minVal) minVal = input[idx];
        }
    }
    output[y * stride + x] = minVal;
}

```

```cpp
// GPU_Launchers.cpp (NativeCore 내부 호출)
#include "GPU_Kernels.cuh"

void LaunchBinarizationKernel(unsigned char* data, int width, int height, int stride, unsigned char threshold) {
    unsigned char* d_data;
    cudaMalloc(&d_data, height * stride);
    cudaMemcpy(d_data, data, height * stride, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    BinarizationKernel<<<grid, block>>>(d_data, width, height, stride, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, height * stride, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// 필터, Dilation/Erosion도 비슷한 구조로 Launch 함수 구현

```

## sobel

```cpp
__global__ void SobelKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride,
                            const double* kernelX, const double* kernelY, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int center = kernelSize / 2;

    if (x < center || y < center || x >= width - center || y >= height - center) return;

    double sumX = 0.0;
    double sumY = 0.0;

    for (int ky = -center; ky <= center; ++ky) {
        for (int kx = -center; kx <= center; ++kx) {
            int idx = (y + ky) * stride + (x + kx);
            int kIdx = (ky + center) * kernelSize + (kx + center);
            sumX += kernelX[kIdx] * input[idx];
            sumY += kernelY[kIdx] * input[idx];
        }
    }

    double val = sqrt(sumX * sumX + sumY * sumY);
    if (val > 255) val = 255;
    output[y * stride + x] = static_cast<unsigned char>(val);
}

```

```cpp
void LaunchSobel(unsigned char* img, int width, int height, int stride,
                 const double* kernelX, const double* kernelY, int kernelSize) {
    unsigned char* d_input; 
    unsigned char* d_output;
    double* d_kX; 
    double* d_kY;

    size_t imgSize = height * stride;
    size_t kernelBytes = kernelSize * kernelSize * sizeof(double);

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_kX, kernelBytes);
    cudaMalloc(&d_kY, kernelBytes);

    cudaMemcpy(d_input, img, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kX, kernelX, kernelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kY, kernelY, kernelBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    SobelKernel<<<grid, block>>>(d_input, d_output, width, height, stride, d_kX, d_kY, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(img, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_kX); cudaFree(d_kY);
}

```

## Gaussian

```cpp
__global__ void GaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride,
                                   const double* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int center = kernelSize / 2;

    if (x < center || y < center || x >= width - center || y >= height - center) return;

    double sum = 0.0;
    for (int ky = -center; ky <= center; ++ky) {
        for (int kx = -center; kx <= center; ++kx) {
            int idx = (y + ky) * stride + (x + kx);
            int kIdx = (ky + center) * kernelSize + (kx + center);
            sum += kernel[kIdx] * input[idx];
        }
    }

    if (sum > 255) sum = 255;
    if (sum < 0) sum = 0;

    output[y * stride + x] = static_cast<unsigned char>(sum);
}

```

## NCC

```cpp
__global__ void NccKernel(const unsigned char* image, const unsigned char* templ, unsigned char* output,
                          int width, int height, int stride,
                          int tempWidth, int tempHeight, int tempStride, double meanT, double stdT) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width - tempWidth || y > height - tempHeight) return;

    double sumI = 0.0;
    double sumI2 = 0.0;
    double crossSum = 0.0;

    for (int ty = 0; ty < tempHeight; ++ty) {
        for (int tx = 0; tx < tempWidth; ++tx) {
            int imgIdx = (y + ty) * stride + (x + tx);
            int tmpIdx = ty * tempStride + tx;
            double diffI = image[imgIdx];
            sumI += diffI;
            sumI2 += diffI * diffI;
        }
    }

    double meanI = sumI / (tempWidth * tempHeight);
    double stdI = sqrt(sumI2 / (tempWidth * tempHeight) - meanI * meanI);

    if (stdI > 0 && stdT > 0) {
        for (int ty = 0; ty < tempHeight; ++ty) {
            for (int tx = 0; tx < tempWidth; ++tx) {
                int imgIdx = (y + ty) * stride + (x + tx);
                int tmpIdx = ty * tempStride + tx;
                crossSum += (image[imgIdx] - meanI) * (templ[tmpIdx] - meanT);
            }
        }
        double ncc = crossSum / (stdI * stdT);
        unsigned char val = static_cast<unsigned char>((ncc + 1.0) * 127.5);
        output[y * stride + x] = val;
    }
    else {
        output[y * stride + x] = 0;
    }
}

```