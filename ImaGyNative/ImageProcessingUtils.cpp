#include "pch.h"
#include "ImageProcessingUtils.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <complex> // std::complex 
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <memory>    // std::unique_ptr 
#include <omp.h>     // OpenMP to CPU Parallel
#include <stdexcept> // std::invalid_argument exception 

const double PI = acos(-1); // math pi use
using Complex = std::complex<double>; // standard colplex library use 

namespace ImaGyNative
{
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
        if (kernelSize % 2 == 0) {
            throw std::invalid_argument("Kernel size must be an odd number.");
        }

        std::vector<double> kernel(kernelSize * kernelSize);
        double sum = 0.0;
        int center = kernelSize / 2;
        double radiusSq = center * center;

#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int x = j - center;
                int y = i - center;

                if (isCircular && (x * x + y * y) > radiusSq) {
                    kernel[i * kernelSize + j] = 0.0;
                    continue;
                }

                double value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
                kernel[i * kernelSize + j] = value;
                sum += value;
            }
        }

        if (sum > 0) {
#pragma omp parallel for
            for (int i = 0; i < kernel.size(); ++i) {
                kernel[i] /= sum;
            }
        }
        return kernel;
    }

    // Average Blur Kernel is not necessary but To reuse convolution function
    std::vector<double> createAverageKernel(int kernelSize, bool isCircular)
    {
        if (kernelSize % 2 == 0) kernelSize++;
        // kernel initialize 
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

    int OtsuThreshold(const unsigned char* sourcePixels, int width, int height, int stride)
    {
        std::vector<int> hist(256, 0);

#pragma omp parallel
        {
            std::vector<int> local_hist(256, 0);
#pragma omp for nowait
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    local_hist[sourcePixels[y * stride + x]]++;
                }
            }
#pragma omp critical
            for (int i = 0; i < 256; ++i) {
                hist[i] += local_hist[i];
            }
        }

        long long total = (long long)width * height;
        double sumAll = 0;
        for (int i = 0; i < 256; i++) {
            sumAll += i * hist[i];
        }

        double sumB = 0;
        long long wB = 0;
        long long wF = 0;
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

    void FFT_1D_Recursive(Complex* data, int N, bool isInverse) {
        if (N <= 1) return;

        auto even = std::make_unique<Complex[]>(N / 2);
        auto odd = std::make_unique<Complex[]>(N / 2);

        for (int i = 0; i < N / 2; ++i) {
            even[i] = data[2 * i];
            odd[i] = data[2 * i + 1];
        }

        FFT_1D_Recursive(even.get(), N / 2, isInverse);
        FFT_1D_Recursive(odd.get(), N / 2, isInverse);

        double angleSign = isInverse ? 1.0 : -1.0;

        for (int k = 0; k < N / 2; ++k) {
            double angle = angleSign * 2.0 * PI * k / N;
            Complex twiddle = { cos(angle), sin(angle) };
            Complex product = twiddle * odd[k];

            data[k] = even[k] + product;
            data[k + N / 2] = even[k] - product;
        }
    }
    void FFT_1D_Iterative(Complex* data, int N, bool isInverse) {
        // 비트 반전(bit-reversal) 순서로 재배열
        int j = 0;
        for (int i = 1; i < N; i++) {
            int bit = N >> 1;
            for (; j & bit; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                Complex temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }

        // iterative 단계 수행
        double angleSign = isInverse ? 1.0 : -1.0;
        for (int len = 2; len <= N; len <<= 1) {
            double angle = angleSign * 2 * PI / len;
            Complex wlen = { cos(angle), sin(angle) };

            #pragma omp parallel for
            for (int i = 0; i < N; i += len) {
                Complex w = { 1.0, 0.0 };
                for (int j = 0; j < len / 2; j++) {
                    Complex u = data[i + j];
                    Complex v = w * data[i + j + len / 2];

                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;

                    w = w * wlen;
                }
            }
        }
        // 역변환 파라미터 확인 후 1/N 스케일링
        if (isInverse) {
#pragma omp parallel for
            for (int i = 0; i < N; i++) {
                data[i] /= N;
            }
        }
    }


    void ApplyFFT2D_CPU(const void* inputPixels, Complex* outputSpectrum, int width, int height, int stride, bool isInverse)
    {
        const unsigned char* pixels = static_cast<const unsigned char*>(inputPixels);
        auto tempComplexData = std::make_unique<Complex[]>(width * height);

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double grayValue = static_cast<double>(pixels[y * stride + x]);
                tempComplexData.get()[y * width + x] = { grayValue, 0.0 };
            }
        }

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            FFT_1D_Iterative(&tempComplexData.get()[y * width], width, isInverse);
        }
        auto all_columns_buffer = std::make_unique<Complex[]>(width * height);

#pragma omp parallel for
        for (int x = 0; x < width; ++x) {
            Complex* tempColumn = &all_columns_buffer.get()[x * height];
            for (int y = 0; y < height; ++y) {
                tempColumn[y] = tempComplexData.get()[y * width + x];
            }
            FFT_1D_Iterative(tempColumn, height, isInverse);
            for (int y = 0; y < height; ++y) {
                tempComplexData.get()[y * width + x] = tempColumn[y];
            }
        }
//
//        if (isInverse) {
//            double scale = 1.0 / (width * height);
//#pragma omp parallel for
//            for (int i = 0; i < width * height; ++i) {
//                tempComplexData[i] = tempComplexData[i] * scale;
//            }
//        }

#pragma omp parallel for
        for (int i = 0; i < width * height; ++i) {
            outputSpectrum[i] = tempComplexData.get()[i];
        }
    }

    void FFT_Shift2D(Complex* spectrum, int width, int height) {
        int halfWidth = width / 2;
        int halfHeight = height / 2;

#pragma omp parallel for
        for (int y = 0; y < halfHeight; ++y) {
            for (int x = 0; x < halfWidth; ++x) {
                std::swap(spectrum[y * width + x], spectrum[(y + halfHeight) * width + (x + halfWidth)]);
            }
            for (int x = halfWidth; x < width; ++x) {
                std::swap(spectrum[y * width + x], spectrum[(y + halfHeight) * width + (x - halfWidth)]);
            }
        }
    }




}