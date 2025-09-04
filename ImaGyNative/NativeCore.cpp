// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include <cmath>

namespace ImaGyNative
{
    // Convolution Helper Method 3x3
    void ApplyConvolution3x3( const unsigned char* sourcePixels, unsigned char* destPixels, 
        int width, int height, int stride, const int kernel[9], double kernelSum)
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

    // // Color Contrast
    // Binarization - Complete
    void NativeCore::ApplyBinarization(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                // Assuming Gray8 format, so each pixel is 1 byte
                int index = y * stride + x;
                pixelData[index] = (pixelData[index] > threshold) ? 255 : 0;
            }
        }
    }
    // Equalization - Notyet
    void NativeCore::ApplyEqualization(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int centerIndex = y * stride + x;
                int indexPy1 = (y + 1) * stride + x;
                int indexPx1 = y * stride + (x + 1);
                int indexPy1Px1 = (y + 1) * stride + x;


                pixelData[centerIndex] = (pixelData[centerIndex] > threshold) ? 255 : 0;
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
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 1, 1,
                         1, 1, 1,
                         1, 1, 1 }; // plane kernel
        double kernelSum = 9.0;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // convolution helper method 
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, kernelSum);

        // Copy the result to origin mem
        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer; // memory free - result buffer
    }
    // Gaussian - Complete
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 2, 1, 
                          2, 4, 2, 
                          1, 2, 1 };
        double kernelSum = 16.0; // blur have to Divide by Sum of Kernel for brightness

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // convolution helper method 
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, kernelSum);

        // Copy the result to origin mem
        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer; // memory free - result buffer
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
    // Image Matching - 여긴 이번주에 못할 듯
    // normailized cross correlation
    void NativeCore::ApplyNCC(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int arr[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 ,1 / 9 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // 1번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); // 배열 0번째
                int indexNxPy = (y + 1) * stride + x; // 배열 1번째
                int indexPxPy = (y + 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); //배열 3번째 
                int indexNxMy = (y - 1) * stride + x; // 배열 4번째
                int indexPxMy = (y - 1) * stride + (x + 1);



                pixelData[indexNxNy];
            }
        }
    }
    void NativeCore::ApplySAD(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int arr[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 ,1 / 9 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // 1번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); // 배열 0번째
                int indexNxPy = (y + 1) * stride + x; // 배열 1번째
                int indexPxPy = (y + 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); //배열 3번째 
                int indexNxMy = (y - 1) * stride + x; // 배열 4번째
                int indexPxMy = (y - 1) * stride + (x + 1);



                pixelData[indexNxNy];
            }
        }
    }
    void NativeCore::ApplySSD(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int arr[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 ,1 / 9 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // 1번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); // 배열 0번째
                int indexNxPy = (y + 1) * stride + x; // 배열 1번째
                int indexPxPy = (y + 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); //배열 3번째 
                int indexNxMy = (y - 1) * stride + x; // 배열 4번째
                int indexPxMy = (y - 1) * stride + (x + 1);



                pixelData[indexNxNy];
            }
        }
    }

}
