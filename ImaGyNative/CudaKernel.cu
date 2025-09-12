#include "CudaKernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cufft.h> // FFT 사용

// --- Helper Macro & Constant Memory ---
#define CUDA_CHECK(err_code) do { cudaError_t _err = (err_code); if (_err != cudaSuccess) { return false; } } while (0)
// --- NEW: cuFFT 라이브러리 전용 에러 체크 매크로 추가 ---
#define CUFFT_CHECK(err_code) do { cufftResult _err = (err_code); if (_err != CUFFT_SUCCESS) { return false; } } while (0)

__constant__ float c_filterKernel[625];
__constant__ float c_sobelKernelX[625];
__constant__ float c_sobelKernelY[625];

// --- CPU Helper Functions (커널 생성) ---
std::vector<float> createGaussianKernelFloat(int kernelSize, double sigma, bool isCircular) {
    if (kernelSize % 2 == 0) kernelSize++;
    std::vector<float> kernel(kernelSize * kernelSize);
    float sum = 0.0f;
    int center = kernelSize / 2;
    double radiusSq = center * center;
    const float M_PI_F = 3.1415926535f;
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            if (isCircular && ((x - center) * (x - center) + (y - center) * (y - center)) > radiusSq) {
                kernel[y * kernelSize + x] = 0.0f;
                continue;
            }
            int dx = x - center;
            int dy = y - center;
            float val = expf(-(dx * dx + dy * dy) / (2.0f * (float)sigma * (float)sigma)) / (2.0f * M_PI_F * (float)sigma * (float)sigma);
            kernel[y * kernelSize + x] = val;
            sum += val;
        }
    }
    if (sum > 0) { for (float& val : kernel) { val /= sum; } }
    return kernel;
}

std::vector<float> createAverageKernelFloat(int kernelSize, bool isCircular) {
    if (kernelSize % 2 == 0) kernelSize++;
    std::vector<float> kernel(kernelSize * kernelSize, 0.0f);
    int center = kernelSize / 2;
    double radiusSq = center * center;
    float count = 0.0f;
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            if (isCircular) {
                if (((x - center) * (x - center) + (y - center) * (y - center)) <= radiusSq) {
                    kernel[y * kernelSize + x] = 1.0f;
                    count++;
                }
            }
            else {
                kernel[y * kernelSize + x] = 1.0f;
                count++;
            }
        }
    }
    if (count > 0) { for (float& val : kernel) { val /= count; } }
    return kernel;
}

void createSobelKernelsFloat(std::vector<float>& kernelX, std::vector<float>& kernelY, int kernelSize) {
    if (kernelSize % 2 == 0) kernelSize++;
    kernelX.assign(kernelSize * kernelSize, 0.0f);
    kernelY.assign(kernelSize * kernelSize, 0.0f);
    int center = kernelSize / 2;
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            int dx = x - center;
            int dy = y - center;
            float denom = (float)(dx * dx + dy * dy);
            if (denom > 0) {
                if (dx != 0) kernelX[y * kernelSize + x] = dx / denom;
                if (dy != 0) kernelY[y * kernelSize + x] = dy / denom;
            }
        }
    }
}

std::vector<float> createLaplacianKernelFloat(int kernelSize) {
    if (kernelSize % 2 == 0) kernelSize++;
    std::vector<float> kernel(kernelSize * kernelSize, -1.0f);
    int centerIndex = (kernelSize / 2) * kernelSize + (kernelSize / 2);
    kernel[centerIndex] = (float)(kernelSize * kernelSize - 1);
    return kernel;
}


// --- CUDA Kernels ---
#define TILE_DIM 16

// 공유 메모리 타일 로딩을 위한 헬퍼 함수
__device__ void loadTile(const unsigned char* input, unsigned char* tile, int width, int height, int stride, int kernelSize) {
    int center = kernelSize / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int i = ty; i < TILE_DIM + 2 * center; i += TILE_DIM) {
        for (int j = tx; j < TILE_DIM + 2 * center; j += TILE_DIM) {
            int loadX = blockIdx.x * TILE_DIM - center + j;
            int loadY = blockIdx.y * TILE_DIM - center + i;
            if (loadX >= 0 && loadX < width && loadY >= 0 && loadY < height) {
                tile[i * (TILE_DIM + 14) + j] = input[loadY * stride + loadX];
            }
            else {
                tile[i * (TILE_DIM + 14) + j] = 0; // Padding
            }
        }
    }
}

__global__ void BinarizationKernel(unsigned char* data, int width, int height, int stride, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    data[idx] = (data[idx] > threshold) ? 255 : 0;
}

__global__ void ConvolutionSharedMemKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride, int kernelSize) {
    __shared__ unsigned char tile[(TILE_DIM + 14) * (TILE_DIM + 14)];
    loadTile(input, tile, width, height, stride, kernelSize);
    __syncthreads();

    int outX = blockIdx.x * TILE_DIM + threadIdx.x;
    int outY = blockIdx.y * TILE_DIM + threadIdx.y;

    if (outX < width && outY < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                sum += c_filterKernel[ky * kernelSize + kx] * tile[(threadIdx.y + ky) * (TILE_DIM + 14) + (threadIdx.x + kx)];
            }
        }
        output[outY * stride + outX] = (unsigned char)fmaxf(0.f, fminf(255.f, sum));
    }
}

__global__ void SobelSharedMemKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride, int kernelSize) {
    __shared__ unsigned char tile[(TILE_DIM + 14) * (TILE_DIM + 14)];
    loadTile(input, tile, width, height, stride, kernelSize);
    __syncthreads();

    int outX = blockIdx.x * TILE_DIM + threadIdx.x;
    int outY = blockIdx.y * TILE_DIM + threadIdx.y;

    if (outX < width && outY < height) {
        float sumX = 0.0f, sumY = 0.0f;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                int kIdx = ky * kernelSize + kx;
                unsigned char pixel_val = tile[(threadIdx.y + ky) * (TILE_DIM + 14) + (threadIdx.x + kx)];
                sumX += c_sobelKernelX[kIdx] * pixel_val;
                sumY += c_sobelKernelY[kIdx] * pixel_val;
            }
        }
        float finalVal = sqrtf(sumX * sumX + sumY * sumY);
        output[outY * stride + outX] = (unsigned char)fmaxf(0.f, fminf(255.f, finalVal));
    }
}

__global__ void DilationSharedMemKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
    __shared__ unsigned char tile[(TILE_DIM + 14) * (TILE_DIM + 14)];
    loadTile(input, tile, width, height, stride, kernelSize);
    __syncthreads();

    int outX = blockIdx.x * TILE_DIM + threadIdx.x;
    int outY = blockIdx.y * TILE_DIM + threadIdx.y;
    int center = kernelSize / 2;

    if (outX < width && outY < height) {
        unsigned char maxVal = 0;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                if (useCircularKernel && ((kx - center) * (kx - center) + (ky - center) * (ky - center) > center * center)) continue;
                unsigned char val = tile[(threadIdx.y + ky) * (TILE_DIM + 14) + (threadIdx.x + kx)];
                if (val > maxVal) maxVal = val;
            }
        }
        output[outY * stride + outX] = maxVal;
    }
}

__global__ void ErosionSharedMemKernel(const unsigned char* input, unsigned char* output, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
    __shared__ unsigned char tile[(TILE_DIM + 14) * (TILE_DIM + 14)];
    loadTile(input, tile, width, height, stride, kernelSize);
    __syncthreads();

    int outX = blockIdx.x * TILE_DIM + threadIdx.x;
    int outY = blockIdx.y * TILE_DIM + threadIdx.y;
    int center = kernelSize / 2;

    if (outX < width && outY < height) {
        unsigned char minVal = 255;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                if (useCircularKernel && ((kx - center) * (kx - center) + (ky - center) * (ky - center) > center * center)) continue;
                unsigned char val = tile[(threadIdx.y + ky) * (TILE_DIM + 14) + (threadIdx.x + kx)];
                if (val < minVal) minVal = val;
            }
        }
        output[outY * stride + outX] = minVal;
    }
}

__global__ void Histogram256Kernel(const unsigned char* data, int width, int height, int stride, unsigned int* hist) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        atomicAdd(&hist[data[y * stride + x]], 1);
    }
}

__global__ void EqualizationKernel(unsigned char* data, int width, int height, int stride, const int* cdf, int cdfMin, int totalPixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    int val = data[idx];
    int newVal = roundf(((float)cdf[val] - cdfMin) / (totalPixels - cdfMin) * 255.0f);
    data[idx] = (unsigned char)fmaxf(0.f, fminf(255.f, (float)newVal));
}


/**
 * @brief FFT 결과를 받아 Magnitude를 계산하고, 로그 스케일링 및 중앙 정렬을 수행합니다.
 * @param fft_input cufftExecR2C의 결과인 복소수 데이터 포인터
 * @param magnitude_output 계산된 실수(float) Magnitude가 저장될 포인터
 * @param width 이미지 너비
 * @param height 이미지 높이
 */
__global__ void FftShiftAndLogMagnitudeKernel(
    const cufftComplex* fft_input,
    float* magnitude_output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 1. 결과 이미지를 중앙 정렬하기 위한 좌표 계산 (FFT Shift)
    int shifted_x = (x + width / 2) % width;
    int shifted_y = (y + height / 2) % height;

    // cuFFT R2C 결과는 너비가 절반+1 크기로 저장되므로, 원본 좌표를 계산
    int complex_width = width / 2 + 1;
    int input_idx = shifted_y * complex_width + (shifted_x % complex_width);

    // 2. Magnitude 계산: sqrt(real^2 + imag^2)
    cufftComplex val = fft_input[input_idx];
    float magnitude = sqrtf(val.x * val.x + val.y * val.y);

    // 3. 로그 스케일 적용 (어두운 부분을 잘 보이게 함)
    magnitude_output[y * width + x] = log10f(1.0f + magnitude);
}

/**
 * @brief float 버퍼를 0~255 범위의 unsigned char 이미지로 정규화합니다.
 */
__global__ void NormalizeFloatToUcharKernel(
    const float* float_input,
    unsigned char* uchar_output,
    int N,
    float min_val,
    float max_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float range = max_val - min_val;
    // 범위가 0이면 픽셀을 0으로 설정하여 0으로 나누기 방지
    if (range <= 1e-6f) {
        uchar_output[i] = 0;
        return;
    }

    // (현재값 - 최소값) / (최대값 - 최소값) * 255
    float normalized_val = 255.0f * (float_input[i] - min_val) / range;
    uchar_output[i] = (unsigned char)fmaxf(0.f, fminf(255.f, normalized_val));
}

__global__ void NccKernel(const unsigned char* image, const unsigned char* templ, float* result,
    int width, int height, int stride,
    int tempWidth, int tempHeight, int tempStride,
    double meanT, double stdDevT_inv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width - tempWidth || y > height - tempHeight) return;

    double sumI = 0.0, sumI2 = 0.0, sumIT = 0.0;
    int N = tempWidth * tempHeight;

    for (int ty = 0; ty < tempHeight; ++ty) {
        for (int tx = 0; tx < tempWidth; ++tx) {
            double pixelI = image[(y + ty) * stride + (x + tx)];
            double pixelT = templ[ty * tempStride + tx];
            sumI += pixelI;
            sumI2 += pixelI * pixelI;
            sumIT += pixelI * pixelT;
        }
    }

    double meanI = sumI / N;
    double stdDevI = sqrt(fmax(0.0, (sumI2 / N) - (meanI * meanI)));
    if (stdDevI < 1e-6) { // 부동소수점 비교를 위해 작은 값 사용
        result[y * width + x] = -1.0f;
        return;
    }

    double ncc = (sumIT - N * meanI * meanT) * stdDevT_inv / stdDevI;
    result[y * width + x] = (float)ncc;
}

__global__ void UcharToFloatKernel(const unsigned char* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = (float)input[i];
}
__global__ void ComplexMultiplyKernel(cufftComplex* a, const cufftComplex* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float real_a = a[i].x;
        float imag_a = a[i].y;
        float real_b = b[i].x;
        float imag_b = b[i].y;
        a[i].x = real_a * real_b - imag_a * imag_b;
        a[i].y = real_a * imag_b + imag_a * real_b;
    }
}
__global__ void NormalizeAndConvertToUcharKernel(const float* input, unsigned char* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = input[i] / N; // IFFT 결과는 N으로 나눠서 정규화
        output[i] = (unsigned char)fmaxf(0.f, fminf(255.f, val));
    }
}


// --- Launcher Functions ---

bool LaunchBinarizationKernel(unsigned char* pixels, int width, int height, int stride, int threshold) {
    unsigned char* d_pixels = nullptr;
    size_t imageSize = (size_t)height * stride;
    CUDA_CHECK(cudaMalloc(&d_pixels, imageSize));
    CUDA_CHECK(cudaMemcpy(d_pixels, pixels, imageSize, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    BinarizationKernel << <grid, block >> > (d_pixels, width, height, stride, (unsigned char)threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_pixels, imageSize, cudaMemcpyDeviceToHost));
    cudaFree(d_pixels);
    return true;
}

bool LaunchEqualizationKernel(unsigned char* pixels, int width, int height, int stride) {
    unsigned char* d_pixels = nullptr;
    unsigned int* d_hist = nullptr;
    int* d_cdf = nullptr;
    size_t imageSize = (size_t)height * stride;
    size_t histSize = 256 * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_pixels, imageSize));
    CUDA_CHECK(cudaMalloc(&d_hist, histSize));
    CUDA_CHECK(cudaMemcpy(d_pixels, pixels, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, histSize));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    Histogram256Kernel << <grid, block >> > (d_pixels, width, height, stride, d_hist);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> h_hist(256);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, histSize, cudaMemcpyDeviceToHost));

    std::vector<int> h_cdf(256);
    h_cdf[0] = h_hist[0];
    for (int i = 1; i < 256; ++i) h_cdf[i] = h_cdf[i - 1] + h_hist[i];

    int cdfMin = 0;
    for (int i = 0; i < 256; ++i) {
        if (h_cdf[i] > 0) {
            cdfMin = h_cdf[i];
            break;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_cdf, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cdf, h_cdf.data(), 256 * sizeof(int), cudaMemcpyHostToDevice));

    EqualizationKernel << <grid, block >> > (d_pixels, width, height, stride, d_cdf, cdfMin, width * height);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(pixels, d_pixels, imageSize, cudaMemcpyDeviceToHost));
    cudaFree(d_pixels);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    return true;
}

bool LaunchGaussianBlurKernel(unsigned char* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel) {
    if (kernelSize > 15) return false;
    unsigned char* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = (size_t)height * stride;
    std::vector<float> h_kernel = createGaussianKernelFloat(kernelSize, sigma, useCircularKernel);

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filterKernel, h_kernel.data(), h_kernel.size() * sizeof(float)));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    ConvolutionSharedMemKernel << <grid, block >> > (d_input, d_output, width, height, stride, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

bool LaunchAverageBlurKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
    if (kernelSize > 15) return false;
    unsigned char* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = (size_t)height * stride;
    std::vector<float> h_kernel = createAverageKernelFloat(kernelSize, useCircularKernel);

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filterKernel, h_kernel.data(), h_kernel.size() * sizeof(float)));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    ConvolutionSharedMemKernel << <grid, block >> > (d_input, d_output, width, height, stride, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

bool LaunchSobelKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize) {
    if (kernelSize > 15) return false;
    unsigned char* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = (size_t)height * stride;
    std::vector<float> h_kernelX, h_kernelY;
    createSobelKernelsFloat(h_kernelX, h_kernelY, kernelSize);

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelKernelX, h_kernelX.data(), h_kernelX.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelKernelY, h_kernelY.data(), h_kernelY.size() * sizeof(float)));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    SobelSharedMemKernel << <grid, block >> > (d_input, d_output, width, height, stride, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

bool LaunchLaplacianKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize) {
    if (kernelSize > 15) return false;
    unsigned char* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = (size_t)height * stride;
    std::vector<float> h_kernel = createLaplacianKernelFloat(kernelSize);

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filterKernel, h_kernel.data(), h_kernel.size() * sizeof(float)));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    ConvolutionSharedMemKernel << <grid, block >> > (d_input, d_output, width, height, stride, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

bool LaunchDilationKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
    if (kernelSize > 15) return false;
    unsigned char* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = (size_t)height * stride;

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    DilationSharedMemKernel << <grid, block >> > (d_input, d_output, width, height, stride, kernelSize, useCircularKernel);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

bool LaunchErosionKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel) {
    if (kernelSize > 15) return false;
    unsigned char* d_input = nullptr, * d_output = nullptr;
    size_t imageSize = (size_t)height * stride;

    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, pixels, imageSize, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    ErosionSharedMemKernel << <grid, block >> > (d_input, d_output, width, height, stride, kernelSize, useCircularKernel);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(pixels, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

bool LaunchNccKernel(const unsigned char* image, int width, int height, int stride,
    const unsigned char* templ, int tempWidth, int tempHeight, int tempStride,
    int* out_x, int* out_y) {
    // 1. CPU에서 템플릿의 평균과 표준편차 계산
    double sumT = 0.0, sumT2 = 0.0;
    int N = tempWidth * tempHeight;
    for (int y = 0; y < tempHeight; ++y) {
        for (int x = 0; x < tempWidth; ++x) {
            sumT += templ[y * tempStride + x];
            sumT2 += templ[y * tempStride + x] * templ[y * tempStride + x];
        }
    }
    double meanT = sumT / N;
    double stdDevT = sqrt(fmax(0.0, (sumT2 / N) - (meanT * meanT)));
    if (stdDevT < 1e-6) return false;

    // 2. GPU 메모리 할당 및 데이터 복사
    unsigned char* d_image = nullptr, * d_templ = nullptr;
    float* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image, (size_t)height * stride));
    CUDA_CHECK(cudaMalloc(&d_templ, (size_t)tempHeight * tempStride));
    CUDA_CHECK(cudaMalloc(&d_result, (size_t)width * height * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_image, image, (size_t)height * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_templ, templ, (size_t)tempHeight * tempStride, cudaMemcpyHostToDevice));

    // 3. NCC 커널 실행
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    NccKernel << <grid, block >> > (d_image, d_templ, d_result, width, height, stride, tempWidth, tempHeight, tempStride, meanT, 1.0 / stdDevT);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 결과를 CPU로 복사하여 최대값 찾기
    std::vector<float> h_result(width * height);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, (size_t)width * height * sizeof(float), cudaMemcpyDeviceToHost));

    float maxNcc = -2.0f;
    *out_x = 0; *out_y = 0;
    for (int y = 0; y <= height - tempHeight; ++y) {
        for (int x = 0; x <= width - tempWidth; ++x) {
            if (h_result[y * width + x] > maxNcc) {
                maxNcc = h_result[y * width + x];
                *out_x = x;
                *out_y = y;
            }
        }
    }

    // 5. 메모리 해제
    cudaFree(d_image);
    cudaFree(d_templ);
    cudaFree(d_result);
    return true;
}

bool LaunchFftFilterKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize) {
    if (width != stride) return false;

    const int N = width * height;
    const int complexWidth = (width / 2 + 1);
    const std::vector<float>& filterKernel = createGaussianKernelFloat(kernelSize, 2, false);
    cufftHandle planR2C, planC2R;
    float* d_input_float = nullptr, * d_kernel_float = nullptr;
    cufftComplex* d_input_complex = nullptr, * d_kernel_complex = nullptr;

    // --- MODIFIED: cufft... 함수에 CUFFT_CHECK 적용 ---
    CUFFT_CHECK(cufftPlan2d(&planR2C, height, width, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&planC2R, height, width, CUFFT_C2R));

    std::vector<float> h_padded_kernel(N, 0.0f);
    int center = kernelSize / 2;
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            int shifted_y = (y - center + height) % height;
            int shifted_x = (x - center + width) % width;
            h_padded_kernel[shifted_y * width + shifted_x] = filterKernel[y * kernelSize + x];
        }
    }

    CUDA_CHECK(cudaMalloc(&d_input_float, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel_float, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_complex, complexWidth * height * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_kernel_complex, complexWidth * height * sizeof(cufftComplex)));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    UcharToFloatKernel << <grid, block >> > (pixels, d_input_float, N);

    CUDA_CHECK(cudaMemcpy(d_kernel_float, h_padded_kernel.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --- MODIFIED: cufft... 함수에 CUFFT_CHECK 적용 ---
    CUFFT_CHECK(cufftExecR2C(planR2C, d_input_float, d_input_complex));
    CUFFT_CHECK(cufftExecR2C(planR2C, d_kernel_float, d_kernel_complex));

    dim3 complex_grid((complexWidth * height + block.x - 1) / block.x);
    ComplexMultiplyKernel << <complex_grid, block >> > (d_input_complex, d_kernel_complex, complexWidth * height);

    // --- MODIFIED: cufft... 함수에 CUFFT_CHECK 적용 ---
    CUFFT_CHECK(cufftExecC2R(planC2R, d_input_complex, d_input_float));

    NormalizeAndConvertToUcharKernel << <grid, block >> > (d_input_float, pixels, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- MODIFIED: cufft... 함수에 CUFFT_CHECK 적용 ---
    CUFFT_CHECK(cufftDestroy(planR2C));
    CUFFT_CHECK(cufftDestroy(planC2R));
    cudaFree(d_input_float);
    cudaFree(d_kernel_float);
    cudaFree(d_input_complex);
    cudaFree(d_kernel_complex);

    return true;
}

/**
 * @brief 이미지의 FFT 스펙트럼을 계산하여 시각적인 이미지로 변환합니다.
 * @param pixels 입출력 버퍼. 원본 이미지가 들어가고, 결과 스펙트럼 이미지로 덮어쓰입니다.
 * @param width 이미지 너비
 * @param height 이미지 높이
 * @param stride 이미지 스트라이드 (width와 같아야 함)
 * @return 성공 시 true, 실패 시 false
 */
bool LaunchFftSpectrumKernel(unsigned char* pixels, int width, int height, int stride) {
    if (width != stride) return false; // 패딩 없는 이미지만 지원

    const int N = width * height;
    const int complexWidth = (width / 2 + 1);

    cufftHandle planR2C;
    float* d_input_float = nullptr;
    float* d_magnitude_float = nullptr;
    cufftComplex* d_input_complex = nullptr;

    // --- 1. 사전 준비 ---
    CUFFT_CHECK(cufftPlan2d(&planR2C, height, width, CUFFT_R2C));
    CUDA_CHECK(cudaMalloc(&d_input_float, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_magnitude_float, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_complex, complexWidth * height * sizeof(cufftComplex)));

    dim3 grid((width + 15) / 16, (height + 15) / 16);
    dim3 block(16, 16);

    // --- 2. 원본 이미지를 GPU의 float 버퍼로 변환 ---
    dim3 grid1D((N + 255) / 256);
    dim3 block1D(256);
    UcharToFloatKernel << <grid1D, block1D >> > (pixels, d_input_float, N);

    // --- 3. FFT 수행 (실수 -> 복소수) ---
    CUFFT_CHECK(cufftExecR2C(planR2C, d_input_float, d_input_complex));

    // --- 4. FFT 결과를 Magnitude 스펙트럼으로 변환 ---
    FftShiftAndLogMagnitudeKernel << <grid, block >> > (d_input_complex, d_magnitude_float, width, height);

    // --- 5. 정규화를 위해 CPU에서 Min/Max 값 찾기 ---
    std::vector<float> h_magnitude(N);
    CUDA_CHECK(cudaMemcpy(h_magnitude.data(), d_magnitude_float, N * sizeof(float), cudaMemcpyDeviceToHost));

    // std::minmax_element를 사용하여 최소/최대값 동시에 찾기
    auto result_pair = std::minmax_element(h_magnitude.begin(), h_magnitude.end());
    auto min_it = result_pair.first;
    auto max_it = result_pair.second;

    float min_val = *min_it;
    float max_val = *max_it;   

    // --- 6. Magnitude를 0~255 범위의 이미지로 정규화 ---
    NormalizeFloatToUcharKernel << <grid1D, block1D >> > (d_magnitude_float, pixels, N, min_val, max_val);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 7. 자원 해제 ---
    CUFFT_CHECK(cufftDestroy(planR2C));
    cudaFree(d_input_float);
    cudaFree(d_magnitude_float);
    cudaFree(d_input_complex);

    return true;
}