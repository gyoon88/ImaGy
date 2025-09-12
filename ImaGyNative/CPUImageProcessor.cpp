#include "pch.h"
#include "NativeCore.h"
#include "ImageProcessingUtils.h"
#include "CPUImageProcessor.h"
#include "CudaKernel.cuh" // Include our new CUDA header
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 
#include <numeric>
#include <algorithm>
#include <stdexcept> // 예외 처리를 위해 추가
#include <cuda_runtime.h>

// Convolution Helper Method
void ApplyConvolution(const unsigned char* sourcePixels, unsigned char* destPixels,
    int width, int height, int stride, const std::vector<double>& kernel, int kernelSize)
{
    int center = kernelSize / 2;
    double kernelSum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    if (kernelSum == 0) kernelSum = 1.0; // 0으로 나누기 방지

    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            double sum = 0.0;
            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    int kernelIndex = (ky + center) * kernelSize + (kx + center);
                    if (kernel[kernelIndex] == 0) continue; // 원형 커널 최적화

                    int sourceIndex = (y + ky) * stride + (x + kx);
                    sum += kernel[kernelIndex] * sourcePixels[sourceIndex];
                }
            }

            // 정규화된 가우시안 커널의 합은 1에 가까우므로 나눌 필요가 없지만, 평균 필터를 위해 필요.
            double finalValue = (kernelSum == 1.0) ? sum : sum / kernelSum;

            if (finalValue > 255) finalValue = 255;
            if (finalValue < 0) finalValue = 0;
            destPixels[y * stride + x] = static_cast<unsigned char>(finalValue);
        }
    }
}

// 32비트 BGRA 컬러 컨볼루션
void ApplyConvolutionColor(const unsigned char* sourcePixels, unsigned char* destPixels,
    int width, int height, int stride, const std::vector<double>& kernel, int kernelSize)
{
    int center = kernelSize / 2;
    double kernelSum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    if (kernelSum == 0) kernelSum = 1.0;

    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            double sumB = 0.0, sumG = 0.0, sumR = 0.0;

            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    int kernelIndex = (ky + center) * kernelSize + (kx + center);
                    if (kernel[kernelIndex] == 0) continue;

                    int sourcePixelX = x + kx;
                    int sourcePixelY = y + ky;
                    const unsigned char* p = sourcePixels + sourcePixelY * stride + sourcePixelX * 4;

                    sumB += kernel[kernelIndex] * p[0]; // B
                    sumG += kernel[kernelIndex] * p[1]; // G
                    sumR += kernel[kernelIndex] * p[2]; // R
                }
            }

            double finalB = (kernelSum == 1.0) ? sumB : sumB / kernelSum;
            double finalG = (kernelSum == 1.0) ? sumG : sumG / kernelSum;
            double finalR = (kernelSum == 1.0) ? sumR : sumR / kernelSum;

            unsigned char* destP = destPixels + y * stride + x * 4;
            destP[0] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, finalB)));
            destP[1] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, finalG)));
            destP[2] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, finalR)));
            // Alpha 채널은 원본 값 그대로 복사
            const unsigned char* srcP = sourcePixels + y * stride + x * 4;
            destP[3] = srcP[3];
        }
    }
}


// Binarization - Complete
void ApplyBinarization_CPU(void* pixels, int width, int height, int stride, int threshold)
{
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    if (threshold == -1) {
        threshold = threshold = OtsuThreshold(pixelData, width, height, stride);
    }
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int index = y * stride + x;
            pixelData[index] = (pixelData[index] > threshold) ? 255 : 0;
        }
    }
}

// Equalization - Complete
void ApplyEqualization_CPU(void* pixels, int width, int height, int stride, unsigned char threshold)
{
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    long long histogram[256] = { 0 }; // Calcuate the Distibution
    long long cdf[256] = { 0 };
    long long totalPixels = width * height;

    // Calculate Histogram
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            histogram[pixelData[y * stride + x]]++;
        }
    }

    // Calculate Cumulative Distribution Function (CDF)
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i)
    {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Find the first non-zero CDF value
    long long cdf_min = 0;
    for (int i = 0; i < 256; ++i)
    {
        if (cdf[i] > 0)
        {
            cdf_min = cdf[i];
            break;
        }
    }

    // Apply Mapping
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int originalPixelValue = pixelData[y * stride + x];
            // Apply the equalization formula
            int newPixelValue = round(((double)cdf[originalPixelValue] - cdf_min) / (totalPixels - cdf_min) * 255.0);

            // Clamp values to 0-255 range
            if (newPixelValue < 0) newPixelValue = 0;
            if (newPixelValue > 255) newPixelValue = 255;

            pixelData[y * stride + x] = static_cast<unsigned char>(newPixelValue);
        }
    }
}


// Differential - Complete
void ApplyDifferential_CPU(void* pixels, int width, int height, int stride, unsigned char threshold)
{
    // origin data 
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    // newbuffer for return 
    unsigned char* resultBuffer = new unsigned char[height * stride];

    for (int y = 0; y < height - 1; ++y)
    {
        for (int x = 0; x < width - 1; ++x)
        {
            int centerIndex = y * stride + x;
            int indexPx = y * stride + (x + 1);
            int indexPy = (y + 1) * stride + x;

            // Calculate Diff each axis
            int gradX = pixelData[indexPx] - pixelData[centerIndex];
            int gradY = pixelData[indexPy] - pixelData[centerIndex];

            // absolute value for velocity
            int val = abs(gradX) + abs(gradY); // val never under 0

            // value validation
            if (val > 255) val = 255;
            unsigned char finalValue = val;

            resultBuffer[centerIndex] = finalValue;
        }
    }

    // copy the result To holding memory address
    memcpy(pixelData, resultBuffer, height * stride); // memcpy(hold memory address, change content address, size) 

    // free the resultBuffer memory
    delete[] resultBuffer;
}

// Sobel - Complete
void ApplySobel_CPU(void* pixels, int width, int height, int stride, int kernelSize)
{
    // 커널 크기는 홀수
    if (kernelSize % 2 == 0) kernelSize++;

    std::vector<double> kernelX = createSobelKernelX(kernelSize);
    std::vector<double> kernelY = createSobelKernelY(kernelSize);

    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* sourceBuffer = new unsigned char[height * stride];
    memcpy(sourceBuffer, pixelData, height * stride);

    // Gx와 Gy 결과를 저장할 임시 버퍼
    double* bufferX = new double[height * stride]();
    double* bufferY = new double[height * stride]();

    int center = kernelSize / 2;

    // Gx와 Gy를 각각 계산
    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            double sumX = 0.0;
            double sumY = 0.0;
            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    int sourceIndex = (y + ky) * stride + (x + kx);
                    int kernelIndex = (ky + center) * kernelSize + (kx + center);
                    sumX += kernelX[kernelIndex] * sourceBuffer[sourceIndex];
                    sumY += kernelY[kernelIndex] * sourceBuffer[sourceIndex];
                }
            }
            int destIndex = y * stride + x;
            bufferX[destIndex] = sumX;
            bufferY[destIndex] = sumY;
        }
    }

    // 결과 병합
    for (int i = 0; i < height * stride; ++i) {
        double finalValue = sqrt(bufferX[i] * bufferX[i] + bufferY[i] * bufferY[i]);
        if (finalValue > 255) finalValue = 255;
        pixelData[i] = static_cast<unsigned char>(finalValue);
    }

    delete[] sourceBuffer;
    delete[] bufferX;
    delete[] bufferY;
}

// Laplacian - Complete
void ApplyLaplacian_CPU(void* pixels, int width, int height, int stride, int kernelSize)
{
    // 커널 크기 검증 
    if (kernelSize % 2 == 0) kernelSize++;

    std::vector<double> kernel = createLaplacianKernel(kernelSize);

    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* resultBuffer = new unsigned char[height * stride];
    memcpy(resultBuffer, pixelData, height * stride);

    // 일반화된 컨볼루션 함수 호출 (kernelSum = 0으로 하여 엣지 강조)
    ApplyConvolution(pixelData, resultBuffer, width, height, stride, kernel, kernelSize);

    memcpy(pixelData, resultBuffer, height * stride);
    delete[] resultBuffer;
}

// // Blurring
// Gaussian - Complete
void ApplyGaussianBlur_CPU(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
{
    std::vector<double> kernel = createGaussianKernel(kernelSize, sigma, useCircularKernel);
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* resultBuffer = new unsigned char[height * stride];
    memcpy(resultBuffer, pixelData, height * stride);

    ApplyConvolution(pixelData, resultBuffer, width, height, stride, kernel, kernelSize);

    memcpy(pixelData, resultBuffer, height * stride);
    delete[] resultBuffer;
}


// 평균 블러
void ApplyAverageBlur_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
{
    std::vector<double> kernel = createAverageKernel(kernelSize, useCircularKernel);
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* resultBuffer = new unsigned char[height * stride];
    memcpy(resultBuffer, pixelData, height * stride);

    ApplyConvolution(pixelData, resultBuffer, width, height, stride, kernel, kernelSize);

    memcpy(pixelData, resultBuffer, height * stride);
    delete[] resultBuffer;
}


// Morphorogy
// 팽창 
void ApplyDilation_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
{
    if (kernelSize % 2 == 0) kernelSize++;
    int center = kernelSize / 2;
    double radiusSq = center * center;

    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* sourceBuffer = new unsigned char[height * stride];
    memcpy(sourceBuffer, pixelData, height * stride);

    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            unsigned char maxValue = 0;
            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    if (useCircularKernel && (kx * kx + ky * ky) > radiusSq) {
                        continue;
                    }
                    unsigned char currentVal = sourceBuffer[(y + ky) * stride + (x + kx)];
                    if (currentVal > maxValue) {
                        maxValue = currentVal;
                    }
                }
            }
            pixelData[y * stride + x] = maxValue;
        }
    }
    delete[] sourceBuffer;
}

// 침식
void ApplyErosion_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
{
    if (kernelSize % 2 == 0) kernelSize++;
    int center = kernelSize / 2;
    double radiusSq = center * center;

    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* sourceBuffer = new unsigned char[height * stride];
    memcpy(sourceBuffer, pixelData, height * stride);

    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            unsigned char minValue = 255;
            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    if (useCircularKernel && (kx * kx + ky * ky) > radiusSq) {
                        continue;
                    }
                    unsigned char currentVal = sourceBuffer[(y + ky) * stride + (x + kx)];
                    if (currentVal < minValue) {
                        minValue = currentVal;
                    }
                }
            }
            pixelData[y * stride + x] = minValue;
        }
    }
    delete[] sourceBuffer;
}

// Image Matching 
// normailized cross correlation
void ApplyNCC_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight,
    int templateStride, int* outCoords)
{
    unsigned char* sourceBuffer = static_cast<unsigned char*>(pixels);
    unsigned char* templateBuffer = static_cast<unsigned char*>(templatePixels);

    double maxNccValue = -2.0;
    int bestX = 0;
    int bestY = 0;

    long long templatePixelCount = (long long)templateWidth * templateHeight;

    // Calculate the mean of the template
    double templateSum = 0.0;
    for (int ty = 0; ty < templateHeight; ++ty)
    {
        for (int tx = 0; tx < templateWidth; ++tx)
        {
            templateSum += templateBuffer[ty * templateStride + tx];
        }
    }
    double meanT = templateSum / templatePixelCount;

    // Calculate sum of squared differences from the mean for the template
    double templateSqDiffSum = 0.0;
    for (int ty = 0; ty < templateHeight; ++ty)
    {
        for (int tx = 0; tx < templateWidth; ++tx)
        {
            double diff = templateBuffer[ty * templateStride + tx] - meanT;
            templateSqDiffSum += diff * diff;
        }
    }

    // Iterate over the source image
    for (int y = 0; y <= height - templateHeight; ++y)
    {
        for (int x = 0; x <= width - templateWidth; ++x)
        {
            double patchSum = 0.0;
            for (int py = 0; py < templateHeight; ++py)
            {
                for (int px = 0; px < templateWidth; ++px)
                {
                    patchSum += sourceBuffer[(y + py) * stride + (x + px)];
                }
            }
            double meanI = patchSum / templatePixelCount;

            double patchSqDiffSum = 0.0;
            double crossCorrelationSum = 0.0;
            for (int ty = 0; ty < templateHeight; ++ty)
            {
                for (int tx = 0; tx < templateWidth; ++tx)
                {
                    double imagePixel = sourceBuffer[(y + ty) * stride + (x + tx)];
                    double templatePixel = templateBuffer[ty * templateStride + tx];

                    double imageDiff = imagePixel - meanI;
                    double templateDiff = templatePixel - meanT;

                    patchSqDiffSum += imageDiff * imageDiff;
                    crossCorrelationSum += imageDiff * templateDiff;
                }
            }

            double denominator = sqrt(patchSqDiffSum * templateSqDiffSum);

            double nccValue = 0.0;
            if (denominator > 0)
            {
                nccValue = crossCorrelationSum / denominator;
            }

            if (nccValue > maxNccValue)
            {
                maxNccValue = nccValue;
                bestX = x;
                bestY = y;
            }
        }
    }
    outCoords[0] = bestX;
    outCoords[1] = bestY;
}



void ApplySAD_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
{
    unsigned char* sourceData = static_cast<unsigned char*>(pixels);
    unsigned char* templateData = static_cast<unsigned char*>(templatePixels);

    double minSadValue = -1.0;
    int bestX = 0;
    int bestY = 0;

    for (int y = 0; y <= height - templateHeight; ++y)
    {
        for (int x = 0; x <= width - templateWidth; ++x)
        {
            double currentSAD = 0.0;
            for (int ty = 0; ty < templateHeight; ++ty)
            {
                for (int tx = 0; tx < templateWidth; ++tx)
                {
                    double imagePixel = sourceData[(y + ty) * stride + (x + tx)];
                    double templatePixel = templateData[ty * templateStride + tx];
                    currentSAD += abs(imagePixel - templatePixel);
                }
            }

            if (minSadValue == -1.0 || currentSAD < minSadValue)
            {
                minSadValue = currentSAD;
                bestX = x;
                bestY = y;
            }
        }
    }
    outCoords[0] = bestX;
    outCoords[1] = bestY;

}

void ApplySSD_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
{
    unsigned char* sourceData = static_cast<unsigned char*>(pixels);
    unsigned char* templateData = static_cast<unsigned char*>(templatePixels);

    double minSsdValue = -1.0;
    int bestX = 0;
    int bestY = 0;

    for (int y = 0; y <= height - templateHeight; ++y)
    {
        for (int x = 0; x <= width - templateWidth; ++x)
        {
            double currentSSD = 0.0;
            for (int ty = 0; ty < templateHeight; ++ty)
            {
                for (int tx = 0; tx < templateWidth; ++tx)
                {
                    double imagePixel = sourceData[(y + ty) * stride + (x + tx)];
                    double templatePixel = templateData[ty * templateStride + tx];
                    double diff = imagePixel - templatePixel;
                    currentSSD += diff * diff;
                }
            }

            if (minSsdValue == -1.0 || currentSSD < minSsdValue)
            {
                minSsdValue = currentSSD;
                bestX = x;
                bestY = y;
            }
        }
    }
    outCoords[0] = bestX;
    outCoords[1] = bestY;
}


////
// // // 
// Color ONly!!! ##############
void ApplyGaussianBlurColor_CPU(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel)
{
    std::vector<double> kernel = createGaussianKernel(kernelSize, sigma, useCircularKernel);
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* resultBuffer = new unsigned char[height * stride];
    memcpy(resultBuffer, pixelData, height * stride);

    ApplyConvolutionColor(pixelData, resultBuffer, width, height, stride, kernel, kernelSize);

    memcpy(pixelData, resultBuffer, height * stride);
    delete[] resultBuffer;
}

void ApplyAverageBlurColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
{
    std::vector<double> kernel = createAverageKernel(kernelSize, useCircularKernel);
    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* resultBuffer = new unsigned char[height * stride];
    memcpy(resultBuffer, pixelData, height * stride);

    ApplyConvolutionColor(pixelData, resultBuffer, width, height, stride, kernel, kernelSize);

    memcpy(pixelData, resultBuffer, height * stride);
    delete[] resultBuffer;
}

void ApplyDilationColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
{
    if (kernelSize % 2 == 0) kernelSize++;
    int center = kernelSize / 2;
    double radiusSq = center * center;

    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* sourceBuffer = new unsigned char[height * stride];
    memcpy(sourceBuffer, pixelData, height * stride);

    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            unsigned char maxB = 0, maxG = 0, maxR = 0;
            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    if (useCircularKernel && (kx * kx + ky * ky) > radiusSq) {
                        continue;
                    }
                    const unsigned char* p = sourceBuffer + (y + ky) * stride + (x + kx) * 4;
                    if (p[0] > maxB) maxB = p[0];
                    if (p[1] > maxG) maxG = p[1];
                    if (p[2] > maxR) maxR = p[2];
                }
            }
            unsigned char* destP = pixelData + y * stride + x * 4;
            destP[0] = maxB;
            destP[1] = maxG;
            destP[2] = maxR;
            destP[3] = sourceBuffer[y * stride + x * 4 + 3]; // Alpha
        }
    }
    delete[] sourceBuffer;
}

void ApplyErosionColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
{
    if (kernelSize % 2 == 0) kernelSize++;
    int center = kernelSize / 2;
    double radiusSq = center * center;

    unsigned char* pixelData = static_cast<unsigned char*>(pixels);
    unsigned char* sourceBuffer = new unsigned char[height * stride];
    memcpy(sourceBuffer, pixelData, height * stride);

    for (int y = center; y < height - center; ++y) {
        for (int x = center; x < width - center; ++x) {
            unsigned char minB = 255, minG = 255, minR = 255;
            for (int ky = -center; ky <= center; ++ky) {
                for (int kx = -center; kx <= center; ++kx) {
                    if (useCircularKernel && (kx * kx + ky * ky) > radiusSq) {
                        continue;
                    }
                    const unsigned char* p = sourceBuffer + (y + ky) * stride + (x + kx) * 4;
                    if (p[0] < minB) minB = p[0];
                    if (p[1] < minG) minG = p[1];
                    if (p[2] < minR) minR = p[2];
                }
            }
            unsigned char* destP = pixelData + y * stride + x * 4;
            destP[0] = minB;
            destP[1] = minG;
            destP[2] = minR;
            destP[3] = sourceBuffer[y * stride + x * 4 + 3]; // Alpha
        }
    }
    delete[] sourceBuffer;
}