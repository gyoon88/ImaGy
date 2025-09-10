// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip> 


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

/**
 * @brief 2D 가우시안 커널을 생성
 * * @param kernelSize 커널의 한 변의 크기. 반드시 홀수
 * @param sigma 가우시안 분포의 표준편차(sigma). 블러의 강도를 조절
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



namespace ImaGyNative
{
    // Convolution Helper Method
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
        if (threshold == -1) {
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
    // Histogram - Complete
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
    
    
    // // EdgeDetect
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
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, int kernelSize)
    {
        // 커널 크기는 홀수여야 합니다.
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
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, int kernelSize)
    {
        // 커널 크기는 홀수여야 합니다.
        if (kernelSize % 2 == 0) kernelSize++;

        std::vector<double> kernel = createLaplacianKernel(kernelSize);

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];
        memcpy(resultBuffer, pixelData, height * stride);

        // 일반화된 컨볼루션 함수 호출 (kernelSum = 0으로 하여 엣지 강조)
        ApplyConvolution(pixelData, resultBuffer, width, height, stride, kernel.data(), kernelSize, 0.0);

        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
    }

    // // Blurring
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
    // Average - Complete
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

    void NativeCore::ApplyDilation(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (kernelSize % 2 == 0) kernelSize++; // 커널 크기를 홀수로 보정
        int center = kernelSize / 2;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        // 결과 버퍼는 원본 데이터로 초기화하여 가장자리를 보존
        unsigned char* resultBuffer = new unsigned char[height * stride];
        memcpy(resultBuffer, pixelData, height * stride);

        for (int y = center; y < height - center; ++y)
        {
            for (int x = center; x < width - center; ++x)
            {
                unsigned char maxValue = 0;
                // 커널 영역 순회
                for (int ky = -center; ky <= center; ++ky)
                {
                    for (int kx = -center; kx <= center; ++kx)
                    {
                        int currentVal = sourceBuffer[(y + ky) * stride + (x + kx)];
                        if (currentVal > maxValue)
                        {
                            maxValue = currentVal;
                        }
                    }
                }
                resultBuffer[y * stride + x] = maxValue;
            }
        }

        memcpy(pixelData, resultBuffer, height * stride);
        delete[] sourceBuffer;
        delete[] resultBuffer;
    }

    // Erosion - Complete
    void NativeCore::ApplyErosion(void* pixels, int width, int height, int stride, int kernelSize)
    {
        if (kernelSize % 2 == 0) kernelSize++; // 커널 크기를 홀수로 보정
        int center = kernelSize / 2;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        unsigned char* resultBuffer = new unsigned char[height * stride];
        memcpy(resultBuffer, pixelData, height * stride);

        for (int y = center; y < height - center; ++y)
        {
            for (int x = center; x < width - center; ++x)
            {
                unsigned char minValue = 255;
                // 커널 영역 순회
                for (int ky = -center; ky <= center; ++ky)
                {
                    for (int kx = -center; kx <= center; ++kx)
                    {
                        int currentVal = sourceBuffer[(y + ky) * stride + (x + kx)];
                        if (currentVal < minValue)
                        {
                            minValue = currentVal;
                        }
                    }
                }
                resultBuffer[y * stride + x] = minValue;
            }
        }

        memcpy(pixelData, resultBuffer, height * stride);
        delete[] sourceBuffer;
        delete[] resultBuffer;
    }

    // Image Matching 
    // normailized cross correlation
    void NativeCore::ApplyNCC(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight,
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
    void NativeCore::ApplySAD(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
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

    void NativeCore::ApplySSD(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords)
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
}
