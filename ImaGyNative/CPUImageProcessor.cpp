#include "pch.h"
#include "NativeCore.h"
#include "ImageProcessingUtils.h"
#include "CPUImageProcessor.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 
#include <numeric>
#include <algorithm>
#include <random> // C++11 
#include <limits> // double 
#include <omp.h>
#include <immintrin.h> 


namespace ImaGyNative
{
    // Convolution Helper Method
    void ApplyConvolution(const unsigned char* sourcePixels, unsigned char* destPixels,
        int width, int height, int stride, const std::vector<double>& kernel, int kernelSize)
    {
        int center = kernelSize / 2;
        double kernelSum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
        if (kernelSum == 0) kernelSum = 1.0;
        #pragma omp parallel for
        for (int y = center; y < height - center; ++y) {
            for (int x = center; x < width - center; ++x) {
                double sum = 0.0;
                for (int ky = -center; ky <= center; ++ky) {
                    for (int kx = -center; kx <= center; ++kx) {
                        int kernelIndex = (ky + center) * kernelSize + (kx + center);
                        if (kernel[kernelIndex] == 0) continue; 

                        int sourceIndex = (y + ky) * stride + (x + kx);
                        sum += kernel[kernelIndex] * sourcePixels[sourceIndex];
                    }
                }

                double finalValue = (kernelSum == 1.0) ? sum : sum / kernelSum;

                if (finalValue > 255) finalValue = 255;
                if (finalValue < 0) finalValue = 0;
                destPixels[y * stride + x] = static_cast<unsigned char>(finalValue);
            }
        }
    }


    void ApplyConvolutionColor(const unsigned char* sourcePixels, unsigned char* destPixels,
        int width, int height, int stride, const std::vector<double>& kernel, int kernelSize)
    {
        int center = kernelSize / 2;
        double kernelSum = std::accumulate(kernel.begin(), kernel.end(), 0.0); // normalization for bright
        if (kernelSum == 0) kernelSum = 1.0;
        #pragma omp parallel for
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
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                histogram[pixelData[y * stride + x]]++;
            }
        }

        // Calculate Cumulative Distribution Function (CDF)
        cdf[0] = histogram[0];
        #pragma omp parallel for
        for (int i = 1; i < 256; ++i)
        {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        // Find the first non-zero CDF value
        long long cdf_min = 0;
        #pragma omp parallel for
        for (int i = 0; i < 256; ++i)
        {
            if (cdf[i] > 0)
            {
                cdf_min = cdf[i];
                break;
            }
        }

        // Apply Mapping
        #pragma omp parallel for
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
        // 
        if (kernelSize % 2 == 0) kernelSize++;

        std::vector<double> kernelX = createSobelKernelX(kernelSize);
        std::vector<double> kernelY = createSobelKernelY(kernelSize);

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        // Gx Gy 
        double* bufferX = new double[height * stride]();
        double* bufferY = new double[height * stride]();

        int center = kernelSize / 2;
    
        // Gx Gy
        #pragma omp parallel for
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
        #pragma omp parallel for

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
  
        if (kernelSize % 2 == 0) kernelSize++;

        std::vector<double> kernel = createLaplacianKernel(kernelSize);

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];
        memcpy(resultBuffer, pixelData, height * stride);

        // �Ϲ�ȭ�� ������� �Լ� ȣ�� (kernelSum = 0���� �Ͽ� ���� ����)
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


    // Average Blur
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
    // Dilation
    void ApplyDilation_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (kernelSize % 2 == 0) kernelSize++;
        int center = kernelSize / 2;
        double radiusSq = center * center;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        #pragma omp parallel for
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

    // Erosion
    void ApplyErosion_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel)
    {
        if (kernelSize % 2 == 0) kernelSize++;
        int center = kernelSize / 2;
        double radiusSq = center * center;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);
        #pragma omp parallel for
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
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (int ty = 0; ty < templateHeight; ++ty)
        {
            for (int tx = 0; tx < templateWidth; ++tx)
            {
                double diff = templateBuffer[ty * templateStride + tx] - meanT;
                templateSqDiffSum += diff * diff;
            }
        }

        #pragma omp parallel for
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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

    // Color ONly!!! 
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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

    const double PI = acos(-1);
    /// <summary>
    /// FFT Spectrum
    /// </summary>
    void ApplyFFT2DSpectrum_CPU(void* pixels, Complex* outputSpectrum, int width, int height, int stride, bool isInverse) {
        const void* readOnlyPixels = static_cast<const void*>(pixels);

        // FFT 
        ApplyFFT2D_CPU(readOnlyPixels, outputSpectrum, width, height, stride, isInverse);

        // DC 
        FFT_Shift2D(outputSpectrum, width, height);

        // 
        float* magnitudes = new float[width * height];
        float maxMagnitude = 0.0;
        #pragma omp parallel for 
        for (int i = 0; i < width * height; ++i) {
            float mag = std::sqrt(outputSpectrum[i].real * outputSpectrum[i].real + outputSpectrum[i].imag * outputSpectrum[i].imag);
            magnitudes[i] = std::log10(1.0 + mag);
            if (magnitudes[i] > maxMagnitude) {
                maxMagnitude = magnitudes[i];
            }
        }

        unsigned char* resultBuffer = new unsigned char[height * stride];
  
        if (maxMagnitude > 0) {
            #pragma omp parallel for
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = y * width + x;
                    resultBuffer[y * stride + x] = static_cast<unsigned char>((magnitudes[index] / maxMagnitude) * 255.0);
                }
            }
        }
        else {
            memset(resultBuffer, 0, height * stride);
        }

        memcpy(pixels, resultBuffer, height * stride);

        delete[] magnitudes;
        delete[] resultBuffer;
    }

    /// <summary>
    /// FFT Phase 
    /// </summary>
    void ApplyFFT2DPhase_CPU(void* pixels, Complex* outputSpectrum, int width, int height, int stride, bool isInverse) {
        ApplyFFT2D_CPU(pixels, outputSpectrum, width, height, stride, false);
        // 
        unsigned char* destPixels = static_cast<unsigned char*>(pixels);
        #pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = y * width + x;
                double phase = std::atan2(outputSpectrum[index].imag, outputSpectrum[index].real);

                unsigned char phaseValue = static_cast<unsigned char>(((phase + PI) / (2.0 * PI)) * 255.0);
                destPixels[y * stride + x] = phaseValue;
            }
        }
    }

    void ApplyFrequencyFilter_CPU(void* pixels, int width, int height, int stride, FilterType filterType, double radiusRatio) {

        auto spectrum = std::make_unique<Complex[]>(width * height);
        const unsigned char* inputPixels = static_cast<const unsigned char*>(pixels);
        double maxRadius = std::min(width, height) / 2.0;
        double radius = maxRadius * radiusRatio;
#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                spectrum[y * width + x] = { static_cast<double>(inputPixels[y * stride + x]), 0.0 };
            }
        }

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            FFT_1D_Iterative(&spectrum[y * width], width, false);
        }

        
#pragma omp parallel
        {
            auto column_buffer = std::make_unique<Complex[]>(height);
            #pragma omp for
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    column_buffer[y] = spectrum[y * width + x];
                }
                FFT_1D_Iterative(column_buffer.get(), height, false);
                for (int y = 0; y < height; ++y) {
                    spectrum[y * width + x] = column_buffer[y];
                }
            }
        }

        FFT_Shift2D(spectrum.get(), width, height);

        double centerX = width / 2.0;
        double centerY = height / 2.0;
#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = y * width + x;
                double distance = std::sqrt(std::pow(x - centerX, 2) + std::pow(y - centerY, 2));
                double mask = (filterType == FilterType::LowPass)
                    ? ((distance <= radius) ? 1.0 : 0.0)
                    : ((distance > radius) ? 1.0 : 0.0);
                spectrum[index] = spectrum[index] * mask;
            }
        }

        FFT_Shift2D(spectrum.get(), width, height);

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            FFT_1D_Iterative(&spectrum[y * width], width, true);
        }

#pragma omp parallel
        {
            auto column_buffer = std::make_unique<Complex[]>(height);
            #pragma omp for
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    column_buffer[y] = spectrum[y * width + x];
                }
                FFT_1D_Iterative(column_buffer.get(), height, true);
                for (int y = 0; y < height; ++y) {
                    spectrum[y * width + x] = column_buffer[y];
                }
            }
        }

        unsigned char* outputPixels = static_cast<unsigned char*>(pixels);
#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double val = spectrum[y * width + x].real;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                outputPixels[y * stride + x] = static_cast<unsigned char>(val);
            }
        }
    }


    void ApplyAxialBandStopFilter_CPU(void* pixels, int width, int height, int stride,
        double lowFreqRadius, double magnitudeThreshold)
    {
        auto spectrum = std::make_unique<Complex[]>(width * height);
        const unsigned char* inputPixels = static_cast<const unsigned char*>(pixels);
#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                spectrum[y * width + x] = { static_cast<double>(inputPixels[y * stride + x]), 0.0 };
            }
        }

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            FFT_1D_Iterative(&spectrum[y * width], width, false);
        }

#pragma omp parallel
        {
            auto column_buffer = std::make_unique<Complex[]>(height);
#pragma omp for
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) column_buffer[y] = spectrum[y * width + x];
                FFT_1D_Iterative(column_buffer.get(), height, false);
                for (int y = 0; y < height; ++y) spectrum[y * width + x] = column_buffer[y];
            }
        }

        // 대칭 이미지 생성
        FFT_Shift2D(spectrum.get(), width, height);

        // 밴드 스톱 필터 마스크 생성 및 적용
        double centerX = width / 2.0;
        double centerY = height / 2.0;
        //double halfThickness = magnitudeThreshold / 2.0;

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = y * width + x;
                double distFromCenter = std::sqrt(std::pow(x - centerX, 2) + std::pow(y - centerY, 2));

                double mask = 1.0; // 기본적으로 모든 주파수를 통과
                
                if (distFromCenter > lowFreqRadius) {
                    
                    double magnitude = std::sqrt(spectrum[index].real * spectrum[index].real + spectrum[index].imag * spectrum[index].imag);
                    if (log(magnitude) > magnitudeThreshold) {
                        mask = 0.0; 
                    }
                }

                spectrum[index] = spectrum[index] * mask;
            }
        }

        // 역변환 
        FFT_Shift2D(spectrum.get(), width, height);

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            FFT_1D_Iterative(&spectrum[y * width], width, true);
        }

#pragma omp parallel
        {
            auto column_buffer = std::make_unique<Complex[]>(height);
#pragma omp for
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) column_buffer[y] = spectrum[y * width + x];
                FFT_1D_Iterative(column_buffer.get(), height, true);
                for (int y = 0; y < height; ++y) spectrum[y * width + x] = column_buffer[y];
            }
        }

        unsigned char* outputPixels = static_cast<unsigned char*>(pixels);
#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double val = spectrum[y * width + x].real;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                outputPixels[y * stride + x] = static_cast<unsigned char>(val);
            }
        }
    }

    // RGB 
    struct ColorPoint {
        double r, g, b;
    };

    void ApplyKMeansClustering_CPU(void* pixels, int width, int height, int stride, int k, int iteration) {
        if (k <= 0) return;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int numPixels = width * height;

        std::vector<ColorPoint> allPixels(numPixels); // 3차원 벡터 생성
#pragma omp parallel for
        for (int i = 0; i < numPixels; ++i) {
            int y = i / width;
            int x = i % width;
            unsigned char* p = pixelData + y * stride + x * 4;
            allPixels[i] = { (double)p[2], (double)p[1], (double)p[0] }; // R, G, B
        }
        // k-means++ 초기화 
        std::vector<ColorPoint> centroids(k);
        std::mt19937 rng(std::random_device{}());

        // 첫 번째 중심점은 무작위로 선택
        std::uniform_int_distribution<int> dist(0, numPixels - 1);
        centroids[0] = allPixels[dist(rng)];

        std::vector<double> minDistSq(numPixels);

        // 나머지 k-1개의 중심점을 선택
        for (int i = 1; i < k; ++i) {
            double totalDistSq = 0.0;

            // 각 픽셀에 대해, 이미 선택된 중심점들과의 가장 짧은 거리(의 제곱)를 계산
#pragma omp parallel for reduction(+:totalDistSq)
            for (int p_idx = 0; p_idx < numPixels; ++p_idx) {
                double currentMinDistSq = std::numeric_limits<double>::max();
                for (int c_idx = 0; c_idx < i; ++c_idx) {
                    double dr = allPixels[p_idx].r - centroids[c_idx].r;
                    double dg = allPixels[p_idx].g - centroids[c_idx].g;
                    double db = allPixels[p_idx].b - centroids[c_idx].b;
                    double distSq = dr * dr + dg * dg + db * db;
                    if (distSq < currentMinDistSq) {
                        currentMinDistSq = distSq;
                    }
                }
                minDistSq[p_idx] = currentMinDistSq;
                totalDistSq += currentMinDistSq;
            }

            // 거리 제곱에 비례하는 확률로 다음 중심점을 선택 (룰렛 휠 선택 방식)
            std::uniform_real_distribution<double> dist_real(0.0, totalDistSq);
            double randVal = dist_real(rng);
            double cumulativeDist = 0.0;
            for (int p_idx = 0; p_idx < numPixels; ++p_idx) {
                cumulativeDist += minDistSq[p_idx];
                if (cumulativeDist >= randVal) {
                    centroids[i] = allPixels[p_idx];
                    break;
                }
            }
        }
        // k-means++ 

        std::vector<int> assignments(numPixels);
        int maxIterations = iteration;

        for (int iter = 0; iter < maxIterations; ++iter) { // repeat til max iteration 
#pragma omp parallel for
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {                    
                    unsigned char* p = pixelData + y * stride + x * 4;
                    double minDistSq = std::numeric_limits<double>::max();
                    int bestCluster = 0;

                    for (int c = 0; c < k; ++c) {
                        // p[2]=R, p[1]=G, p[0]=B
                        double dr = p[2] - centroids[c].r;
                        double dg = p[1] - centroids[c].g;
                        double db = p[0] - centroids[c].b;
                        double distSq = dr * dr + dg * dg + db * db;

                        if (distSq < minDistSq) {
                            minDistSq = distSq;
                            bestCluster = c;
                        }
                    }
                    assignments[y * width + x] = bestCluster;
                }
            }
            std::vector<ColorPoint> newCentroids(k, { 0.0, 0.0, 0.0 });
            std::vector<int> counts(k, 0);

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int clusterId = assignments[y * width + x];                    
                    unsigned char* p = pixelData + y * stride + x * 4;
                    newCentroids[clusterId].r += p[2]; // R
                    newCentroids[clusterId].g += p[1]; // G
                    newCentroids[clusterId].b += p[0]; // B
                    counts[clusterId]++;
                }
            }
            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    centroids[c].r = newCentroids[c].r / counts[c];
                    centroids[c].g = newCentroids[c].g / counts[c];
                    centroids[c].b = newCentroids[c].b / counts[c];
                }
            }
        }
#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int clusterId = assignments[y * width + x];
                unsigned char* p = pixelData + y * stride + x * 4;
                p[2] = static_cast<unsigned char>(centroids[clusterId].r); // R
                p[1] = static_cast<unsigned char>(centroids[clusterId].g); // G
                p[0] = static_cast<unsigned char>(centroids[clusterId].b); // B
                // p[3] is the alpha value so pass it
            }
        }
    }
    
//    void ApplyKMeansClustering_CPU(void* pixels, int width, int height, int stride, int k, int iteration) {
//        if (k <= 0) return;
//
//        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
//        int numPixels = width * height;
//
//        std::vector<ColorPoint> centroids(k);
//        std::mt19937 rng(std::random_device{}());
//        std::uniform_int_distribution<int> dist(0, numPixels);
//
//        for (int i = 0; i < k; ++i) {
//            int randIdx = dist(rng);
//            int y = randIdx / width;
//            int x = randIdx % width;
//
//            // --- [���� 1] �ȼ��� 4����Ʈ�� �ּ� ��� ---
//            unsigned char* p = pixelData + y * stride + x * 4;
//
//            // Bgra32 �����̹Ƿ� B=p[0], G=p[1], R=p[2] ����
//            // ColorPoint ����ü�� r, g, b ������ ������� ��
//            centroids[i] = { (double)p[2], (double)p[1], (double)p[0] }; // R, G, B 
//        }
//
//        std::vector<int> assignments(numPixels);
//        int maxIterations = iteration;
//
//        for (int iter = 0; iter < maxIterations; ++iter) {
//#pragma omp parallel for
//            for (int y = 0; y < height; ++y) {
//                for (int x = 0; x < width; ++x) {
//                    // --- [���� 2] �ȼ��� 4����Ʈ�� �ּ� ��� ---
//                    unsigned char* p = pixelData + y * stride + x * 4;
//                    double minDistSq = std::numeric_limits<double>::max();
//                    int bestCluster = 0;
//
//                    for (int c = 0; c < k; ++c) {
//                        // p[2]=R, p[1]=G, p[0]=B
//                        double dr = p[2] - centroids[c].r;
//                        double dg = p[1] - centroids[c].g;
//                        double db = p[0] - centroids[c].b;
//                        double distSq = dr * dr + dg * dg + db * db;
//
//                        if (distSq < minDistSq) {
//                            minDistSq = distSq;
//                            bestCluster = c;
//                        }
//                    }
//                    assignments[y * width + x] = bestCluster;
//                }
//            }
//
//            std::vector<ColorPoint> newCentroids(k, { 0.0, 0.0, 0.0 });
//            std::vector<int> counts(k, 0);
//
//            for (int y = 0; y < height; ++y) {
//                for (int x = 0; x < width; ++x) {
//                    int clusterId = assignments[y * width + x];
//                    // --- [���� 3] �ȼ��� 4����Ʈ�� �ּ� ��� ---
//                    unsigned char* p = pixelData + y * stride + x * 4;
//                    newCentroids[clusterId].r += p[2]; // R
//                    newCentroids[clusterId].g += p[1]; // G
//                    newCentroids[clusterId].b += p[0]; // B
//                    counts[clusterId]++;
//                }
//            }
//
//            for (int c = 0; c < k; ++c) {
//                if (counts[c] > 0) {
//                    centroids[c].r = newCentroids[c].r / counts[c];
//                    centroids[c].g = newCentroids[c].g / counts[c];
//                    centroids[c].b = newCentroids[c].b / counts[c];
//                }
//            }
//        }
//
//#pragma omp parallel for
//        for (int y = 0; y < height; ++y) {
//            for (int x = 0; x < width; ++x) {
//                int clusterId = assignments[y * width + x];
//                // --- [���� 4] �ȼ��� 4����Ʈ�� �ּ� ��� ---
//                unsigned char* p = pixelData + y * stride + x * 4;
//                p[2] = static_cast<unsigned char>(centroids[clusterId].r); // R
//                p[1] = static_cast<unsigned char>(centroids[clusterId].g); // G
//                p[0] = static_cast<unsigned char>(centroids[clusterId].b); // B
//                // p[3] 
//            }
//        }
//    }
//


    struct Point5D {
        double r, g, b, x, y;
    };

    void ApplyKMeansClusteringXY_Normalized_CPU(void* pixels, int width, int height, int stride, int k, int iteration) {
        if (k <= 0) return;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int numPixels = width * height;

        // Scaling by Min-Max 
        std::vector<Point5D> normalizedPixels(numPixels);
        double w_minus_1 = width > 1 ? (double)(width - 1) : 1.0;
        double h_minus_1 = height > 1 ? (double)(height - 1) : 1.0;

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                unsigned char* p = pixelData + y * stride + x * 4;
                normalizedPixels[y * width + x] = {
                    p[2] / 255.0,       // R
                    p[1] / 255.0,       // G
                    p[0] / 255.0,       // B
                    x / w_minus_1,      // X
                    y / h_minus_1       // Y
                };
            }
        }

        // K-Means 
        std::vector<Point5D> centroids(k);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, numPixels - 1);

        for (int i = 0; i < k; ++i) {
            centroids[i] = normalizedPixels[dist(rng)];
        }

        std::vector<int> assignments(numPixels);
        int maxIterations = iteration;

        for (int iter = 0; iter < maxIterations; ++iter) {
#pragma omp parallel for
            for (int i = 0; i < numPixels; ++i) {
                double minDistSq = std::numeric_limits<double>::max();
                int bestCluster = 0;
                for (int c = 0; c < k; ++c) {
                    double dr = normalizedPixels[i].r - centroids[c].r;
                    double dg = normalizedPixels[i].g - centroids[c].g;
                    double db = normalizedPixels[i].b - centroids[c].b;
                    double dx = normalizedPixels[i].x - centroids[c].x;
                    double dy = normalizedPixels[i].y - centroids[c].y;
                    double distSq = dr * dr + dg * dg + db * db + dx * dx + dy * dy;

                    if (distSq < minDistSq) {
                        minDistSq = distSq;
                        bestCluster = c;
                    }
                }
                assignments[i] = bestCluster;
            }

            std::vector<Point5D> newCentroids(k, { 0.0, 0.0, 0.0, 0.0, 0.0 });
            std::vector<int> counts(k, 0);
            for (int i = 0; i < numPixels; ++i) {
                int clusterId = assignments[i];
                newCentroids[clusterId].r += normalizedPixels[i].r;
                newCentroids[clusterId].g += normalizedPixels[i].g;
                newCentroids[clusterId].b += normalizedPixels[i].b;
                newCentroids[clusterId].x += normalizedPixels[i].x;
                newCentroids[clusterId].y += normalizedPixels[i].y;
                counts[clusterId]++;
            }
            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    centroids[c] = {
                        newCentroids[c].r / counts[c], newCentroids[c].g / counts[c],
                        newCentroids[c].b / counts[c], newCentroids[c].x / counts[c],
                        newCentroids[c].y / counts[c]
                    };
                }
            }
        }

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int clusterId = assignments[y * width + x];
                unsigned char* p = pixelData + y * stride + x * 4;
                p[2] = static_cast<unsigned char>(centroids[clusterId].r * 255.0); // R
                p[1] = static_cast<unsigned char>(centroids[clusterId].g * 255.0); // G
                p[0] = static_cast<unsigned char>(centroids[clusterId].b * 255.0); // B
            }
        }
    }




}