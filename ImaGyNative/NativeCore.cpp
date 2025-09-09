// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 


/**
 * @brief 2D 가우시안 커널을 생성합니다.
 * * @param kernelSize 커널의 한 변의 크기. 반드시 홀수여야 합니다 (e.g., 3, 5, 7).
 * @param sigma 가우시안 분포의 표준편차(sigma). 블러의 강도를 조절합니다.
 * @return 정규화된 1D 벡터 형태의 가우시안 커널.
 */
std::vector<double> createGaussianKernel(int kernelSize, double sigma)
{
    const double M_PI = 3.14159265358979323846;
    // 커널 크기는 반드시 홀수여야 중앙 픽셀이 존재합니다.
    if (kernelSize % 2 == 0) {
        throw std::invalid_argument("Kernel size must be an odd number.");
    }

    std::vector<double> kernel(kernelSize * kernelSize);
    double sum = 0.0;
    int center = kernelSize / 2;

    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            int x = j - center;
            int y = i - center;

            // 2D 가우시안 공식
            double value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);

            kernel[i * kernelSize + j] = value;
            sum += value;
        }
    }

    // 2. 정규화: 모든 커널 값의 합이 1이 되도록 조정
    for (int i = 0; i < kernel.size(); ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}
int OtsuThreshold(const unsigned char* sourcePixels, int width, int height, int stride)
{
    int hist[256] = { 0 };
    int total = width * height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = sourcePixels[y * stride + x];
            hist[idx]++;
        }
    }

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
namespace ImaGyNative
{
    // Convolution Helper Method 3x3
    void ApplyConvolution3x3( const unsigned char* sourcePixels, unsigned char* destPixels, 
        int width, int height, int stride, int kernel[9], double kernelSum)
    {
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double sum = 0.0;

                // neighbor pixel
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // convolution operation
                for (int i = 0; i < 9; ++i)
                {
                    sum += kernel[i] * sourcePixels[indexes[i]];
                }
                           
                if (kernelSum != 0) {  // If kernelSum is 0, the calling method is edge detect 
                    sum /= kernelSum;
                }

                // Pixel value validation
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                destPixels[indexes[4]] = static_cast<unsigned char>(sum);
            }
        }

    }
    void ApplyConvolution(const unsigned char* sourcePixels, unsigned char* destPixels,
        int width, int height, int stride, const double* kernel, int kernelSize, double kernelSum)
    {
        int center = kernelSize / 2;

        for (int y = center; y < height - center; ++y)
        {
            for (int x = center; x < width - center; ++x)
            {
                double sum = 0.0;
                for (int ky = -center; ky <= center; ++ky)
                {
                    for (int kx = -center; kx <= center; ++kx)
                    {
                        int sourceIndex = (y + ky) * stride + (x + kx);
                        int kernelIndex = (ky + center) * kernelSize + (kx + center);
                        sum += kernel[kernelIndex] * sourcePixels[sourceIndex];
                    }
                }

                if (kernelSum != 0)
                {
                    sum /= kernelSum;
                }

                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                destPixels[y * stride + x] = static_cast<unsigned char>(sum);
            }
        }
    }
    // // Color Contrast
    // Binarization - Complete
    void NativeCore::ApplyBinarization(void* pixels, int width, int height, int stride, int threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        if (threshold != 128) {
            threshold = OtsuThreshold(pixelData, width, height, stride);
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
    void NativeCore::ApplyEqualization(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        long long histogram[256] = { 0 }; // Calcuate the Distibution
        long long cdf[256] = { 0 };
        long long totalPixels = width * height;

        // 1. Calculate Histogram
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                histogram[pixelData[y * stride + x]]++;
            }
        }

        // 2. Calculate Cumulative Distribution Function (CDF)
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

        // 3. Apply Mapping
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
    void NativeCore::ApplyHistogram(void* pixels, int width, int height, int stride, int* hist) {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        std::fill(hist, hist + 256, 0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = pixelData[y * stride + x];
                hist[idx]++;
            }
        }
    }
    // EdgeDetect
    // Differnetial - Complete
    void NativeCore::ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold)
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
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        // Gx kernel 
        int kernelX[9] = { -1, 0, 1, 
                           -2, 0, 2, 
                           -1, 0, 1 };
        // Gy kernel
        int kernelY[9] = { -1, -2, -1, 
                            0, 0, 0, 
                            1, 2, 1 };
        // origin data
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // read only data 
        unsigned char* sourceBuffer = new unsigned char[height * stride];

        //copy the origin data to read only data buffer
        memcpy(sourceBuffer, pixelData, height * stride);

        // Generate temp buffer for Gx, Gy each result 
        double* bufferX = new double[height * stride];
        double* bufferY = new double[height * stride];

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double sumX = 0.0;
                double sumY = 0.0;

                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1), y * stride + x, y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // Convolution operation
                for (int i = 0; i < 9; ++i)
                {
                    sumX += kernelX[i] * sourceBuffer[indexes[i]];
                    sumY += kernelY[i] * sourceBuffer[indexes[i]];
                }

                int centerIndex = y * stride + x;
                bufferX[centerIndex] = sumX;
                bufferY[centerIndex] = sumY;
            }
        }

        // Edit the origin data
        for (int i = 0; i < height * stride; ++i)
        {
            double finalValue = sqrt(bufferX[i] * bufferX[i] + bufferY[i] * bufferY[i]); // If they cause time issue change abs 
            // value validation
            if (finalValue > 255) finalValue = 255;
            pixelData[i] = static_cast<unsigned char>(finalValue);
        }

        // free the readonly buffer and each result buffer GxGy 
        delete[] sourceBuffer;
        delete[] bufferX;
        delete[] bufferY;
    }
    // Laplacian - Complete
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 1, 1, 
                          1, -8, 1, 
                          1, 1, 1 }; 
        // Origin data
        unsigned char* pixelData = static_cast<unsigned char*>(pixels); 
        // result data
        unsigned char* resultBuffer = new unsigned char[height * stride]; 

        // Call the convolusion Helper method
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, 0); 
        memcpy(pixelData, resultBuffer, height * stride); // Copy the result to origin mem

        delete[] resultBuffer; // memory free - result buffer
    }

    // Blurring
   // Gaussian - Complete
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, double sigma, int kernelSize)
    {
        // double 타입 벡터로 가우시안 커널을 받음
        std::vector<double> kernel = createGaussianKernel(kernelSize, sigma);

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // 원본 데이터를 결과 버퍼에 복사하여 가장자리 픽셀들이 보존
        memcpy(resultBuffer, pixelData, height * stride);

        // 일반화된 컨볼루션 함수를 호출
        // 커널이 이미 정규화되었으므로 kernelSum은 1.0
        ApplyConvolution(pixelData, resultBuffer, width, height, stride, kernel.data(), kernelSize, 1.0);

        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
    }
    //
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, int kernelSize)
    {
        // double 타입 커널을 사용하도록 수정 (ApplyConvolution 함수와 호환을 위해)
        std::vector<double> kernel(kernelSize * kernelSize);
        for (int i = 0; i < kernel.size(); i++) { // 루프 변수 초기화 및 범위 수정
            kernel[i] = 1.0;
        }

        double kernelSum = kernelSize * kernelSize; // kernelSum 수정

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];
        memcpy(resultBuffer, pixelData, height * stride);

        // 일반화된 컨볼루션 함수 호출
        ApplyConvolution(pixelData, resultBuffer, width, height, stride, kernel.data(), kernelSize, kernelSum);

        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
    }

    // Morphorogy
    void NativeCore::ApplyDilation(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // result and readonly buffer
        unsigned char* resultBuffer = new unsigned char[height * stride];
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // calculate neighbor pixel 
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };
                // change max neighbor pixel 
                unsigned char maxValue = 0;
                for (int i = 0; i < 9; ++i)
                {
                    if (sourceBuffer[indexes[i]] > maxValue)
                    {
                        maxValue = sourceBuffer[indexes[i]];
                    }
                }

                // store the max value to result buffer
                resultBuffer[indexes[4]] = maxValue;
            }
        }

        // copy the result buffer to origin data
        memcpy(pixelData, resultBuffer, height * stride);

        // memory free - result buffer
        delete[] resultBuffer;
        delete[] sourceBuffer;
    }


    // Erosion - Complete
    void NativeCore::ApplyErosion(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // result and readonly buffer
        unsigned char* resultBuffer = new unsigned char[height * stride];
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // calculate neighbor pixel 
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // change the pixel to min of neighbor pixel
                unsigned char minValue = 255;
                for (int i = 0; i < 9; ++i)
                {
                    if (sourceBuffer[indexes[i]] < minValue)
                    {
                        minValue = sourceBuffer[indexes[i]];
                    }
                }

                // store the min value of neighbor to result buffer
                resultBuffer[indexes[4]] = minValue;
            }
        }

        // copy the result to origin buffer
        memcpy(pixelData, resultBuffer, height * stride);

        // mem free
        delete[] resultBuffer;
        delete[] sourceBuffer;
    }
    // Image Matching - I'll complete next week
    // normailized cross correlation
    void NativeCore::ApplyNCC(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold)
    {
        unsigned char* sourceBuffer = static_cast<unsigned char*>(pixels);
        unsigned char* templateBuffer = static_cast<unsigned char*>(templatePixels);

        // result Buffer
        unsigned char* resultBuffer = new unsigned char[height * stride];
        memset(resultBuffer, 0, height * stride); // initialize to black

        // Calculate the templete image
        double templateSum = 0.0;
        for (int ty = 0; ty < templateHeight; ++ty)
        {
            for (int tx = 0; tx < templateWidth; ++tx)
            {
                templateSum += templateBuffer[ty * templateStride + tx];
            }
        }
        double meanT = templateSum / (templateWidth * templateHeight);

        double templateSqSum = 0.0;
        for (int ty = 0; ty < templateHeight; ++ty)
        {
            for (int tx = 0; tx < templateWidth; ++tx)
            {
                double diff = templateBuffer[ty * templateStride + tx] - meanT;
                templateSqSum += diff * diff;
            }
        }
        double stdT = sqrt(templateSqSum / (templateWidth * templateHeight));

        // loop for matching image area
        for (int y = 0; y <= height - templateHeight; ++y)
        {
            for (int x = 0; x <= width - templateWidth; ++x)
            {
                // Current image patch's mean and std
                double patchSum = 0.0;
                for (int py = 0; py < templateHeight; ++py)
                {
                    for (int px = 0; px < templateWidth; ++px)
                    {
                        patchSum += sourceBuffer[(y + py) * stride + (x + px)];
                    }
                }
                double meanI = patchSum / (templateWidth * templateHeight);

                double patchSqSum = 0.0;
                for (int py = 0; py < templateHeight; ++py)
                {
                    for (int px = 0; px < templateWidth; ++px)
                    {
                        double diff = sourceBuffer[(y + py) * stride + (x + px)] - meanI;
                        patchSqSum += diff * diff;
                    }
                }
                double stdI = sqrt(patchSqSum / (templateWidth * templateHeight));

                double crossCorrelationSum = 0.0;
                for (int ty = 0; ty < templateHeight; ++ty)
                {
                    for (int tx = 0; tx < templateWidth; ++tx)
                    {
                        double imagePixel = sourceBuffer[(y + ty) * stride + (x + tx)];
                        double templatePixel = templateBuffer[ty * templateStride + tx];
                        crossCorrelationSum += (imagePixel - meanI) * (templatePixel - meanT);
                    }
                }

                double nccValue = 0.0;
                if (stdI > 0 && stdT > 0) // ZeroDivision Error exception
                {
                    nccValue = crossCorrelationSum / (stdI * stdT);
                }

                // normalization btween -1 and 1
                unsigned char outputValue = static_cast<unsigned char>((nccValue + 1.0) * 127.5);
                resultBuffer[y * stride + x] = outputValue;
            }
        }

        // Copy the resultData to sourceData
        memcpy(sourceBuffer, resultBuffer, height * stride);

        // free mem
        delete[] resultBuffer;
    }

    void NativeCore::ApplySAD(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold)
    {
        unsigned char* sourceData = static_cast<unsigned char*>(pixels);
        unsigned char* templateData = static_cast<unsigned char*>(templatePixels);

        // 결과 이미지용 버퍼 생성
        unsigned char* resultData = new unsigned char[height * stride];
        memset(resultData, 0, height * stride); // 검은색으로 초기화

        double maxSAD = templateWidth * templateHeight * 255.0; // 가능한 최대 SAD 값 (모든 픽셀이 255만큼 다름)

        // 매칭 영역을 찾기 위해 원본 이미지 반복
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

                // SAD 값 정규화: 0 (완벽한 매칭)에서 maxSAD (최악의 매칭)
                // 0은 흰색 (최고의 매칭), maxSAD는 검은색 (최악의 매칭)으로 0-255에 매핑
                unsigned char outputValue = static_cast<unsigned char>(255.0 * (1.0 - (currentSAD / maxSAD)));
                resultData[y * stride + x] = outputValue;
            }
        }

        // 결과를 원본 픽셀 포인터로 다시 복사
        memcpy(sourceData, resultData, height * stride);

        // 할당된 메모리 해제
        delete[] resultData;
    }

    void NativeCore::ApplySSD(void* pixels, int width, int height, int stride, 
        void* templatePixels, int templateWidth, int templateHeight, int templateStride, unsigned char threshold)
    {
    }
}
