#include "pch.h"
#include "NativeCore.h"
#include "CudaKernel.cuh" // Include our new CUDA header
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 
#include <numeric>
#include <algorithm>
#include <stdexcept> // 예외 처리를 위해 추가
#include <cuda_runtime.h>

namespace ImaGyNative
{
    /**
      * @brief 소벨 X축 커널을 생성
      * @param kernelSize 커널 크기 (홀수).
      * @return double 타입의 1D 벡터 커널.
      */
    std::vector<double> createSobelKernelX(int kernelSize) {
        std::vector<double> kernel(kernelSize * kernelSize);
        int center = kernelSize / 2;
        for (int y = 0; y < kernelSize; ++y) {
            for (int x = 0; x < kernelSize; ++x) {
                if (x == center) {
                    kernel[y * kernelSize + x] = 0;
                }
                else {
                    kernel[y * kernelSize + x] = (x - center) / (double)((x - center) * (x - center) + (y - center) * (y - center));
                }
            }
        }
        return kernel;
    }

    /**
     * @brief 소벨 Y축 커널을 생성
     * @param kernelSize 커널 크기 (홀수).
     * @return double 타입의 1D 벡터 커널.
     */
    std::vector<double> createSobelKernelY(int kernelSize) {
        std::vector<double> kernel(kernelSize * kernelSize);
        int center = kernelSize / 2;
        for (int y = 0; y < kernelSize; ++y) {
            for (int x = 0; x < kernelSize; ++x) {
                if (y == center) {
                    kernel[y * kernelSize + x] = 0;
                }
                else {
                    kernel[y * kernelSize + x] = (y - center) / (double)((x - center) * (x - center) + (y - center) * (y - center));
                }
            }
        }
        return kernel;
    }

    /**
     * @brief 라플라시안 커널을 생성
     * @param kernelSize 커널 크기 (홀수).
     * @return double 타입의 1D 벡터 커널.
     */
    std::vector<double> createLaplacianKernel(int kernelSize)
    {
        if (kernelSize % 2 == 0) {
            throw std::invalid_argument("Kernel size must be an odd number.");
        }
        std::vector<double> kernel(kernelSize * kernelSize, 1.0);
        int centerIndex = (kernelSize / 2) * kernelSize + (kernelSize / 2);
        kernel[centerIndex] = 1.0 - (kernelSize * kernelSize);
        return kernel;
    }


    std::vector<double> createGaussianKernel(int kernelSize, double sigma, bool isCircular)
    {
        const double M_PI = 3.14159265358979323846;
        if (kernelSize % 2 == 0) {
            throw std::invalid_argument("Kernel size must be an odd number.");
        }

        std::vector<double> kernel(kernelSize * kernelSize);
        double sum = 0.0;
        int center = kernelSize / 2;
        double radiusSq = center * center;

        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int x = j - center;
                int y = i - center;

                if (isCircular && (x * x + y * y) > radiusSq) {
                    kernel[i * kernelSize + j] = 0.0;
                    continue;
                }

                double value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                kernel[i * kernelSize + j] = value;
                sum += value;
            }
        }

        if (sum > 0) { // 0으로 나누는 것을 방지
            for (double& val : kernel) {
                val /= sum;
            }
        }
        return kernel;
    }

    // 평균 필터를 위한 (원형) 커널 생성 함수
    std::vector<double> createAverageKernel(int kernelSize, bool isCircular)
    {
        if (kernelSize % 2 == 0) kernelSize++;
        std::vector<double> kernel(kernelSize * kernelSize, 0.0);
        int center = kernelSize / 2;
        double radiusSq = center * center;

        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                if (isCircular) {
                    int x = j - center;
                    int y = i - center;
                    if ((x * x + y * y) <= radiusSq) {
                        kernel[i * kernelSize + j] = 1.0;
                    }
                }
                else {
                    kernel[i * kernelSize + j] = 1.0;
                }
            }
        }
        return kernel;
    }


    /**
     * 오츠 알고리즘으로 임계값 산정
     */
    int OtsuThreshold(const unsigned char* sourcePixels, int width, int height, int stride)
    {
        int hist[256] = { 0 };
        int total = width * height;
        // Hitogram Distribution Calculation
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = sourcePixels[y * stride + x];
                hist[idx]++;
            }
        }
        // Expetation Calculation
        double sumAll = 0;
        for (int i = 0; i < 256; i++) {
            sumAll += i * hist[i];
        }

        double sumB = 0;
        int wB = 0;
        int wF = 0;

        double maxVar = 0;
        int threshold = 0;

        for (int t = 0; t < 256; t++) {
            wB += hist[t];
            if (wB == 0) continue;

            wF = total - wB;
            if (wF == 0) break;

            sumB += (double)(t * hist[t]);

            double mB = sumB / wB;
            double mF = (sumAll - sumB) / wF;
            double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);

            if (varBetween > maxVar) {
                maxVar = varBetween;
                threshold = t;
            }
        }

        return threshold;
    }
}