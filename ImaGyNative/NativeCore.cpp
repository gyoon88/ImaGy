// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include <cmath>

namespace ImaGyNative
{
    // // Color Contrast
    // 이진화 - 완료
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
    // 평활화 - 시작안함.
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
    // 미분 구현완료
    void NativeCore::ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        for (int y = 0; y < height-1; ++y)
        {
            for (int x = 0; x < width-1; ++x)
            {
                int centerIndex = y * stride + x;
                int indexPx = y * stride + (x + 1);
                int indexPy = (y + 1) * stride + x;
                int val = abs(pixelData[centerIndex] * -1 + pixelData[indexPy]) + abs(pixelData[centerIndex] * -1 + pixelData[indexPx]);
                pixelData[centerIndex] = static_cast<unsigned char>(min(val, 255));
               
            }
        }
    }
    // 소벨 시작안함
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int kernelX[9] = { -1, 0, 1, 
                        -2, 0, 2, 
                        -1, 0, 1 };
        int kernelY[9] = { -1, -2, -1, 
                        0, 0, 0, 
                        1, 2, 1 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double resultX = 0.0;
                double resultY = 0.0;
                // 1번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); // 배열 0번째
                int indexNxMy = (y - 1) * stride + x; // 배열 1번째
                int indexPxMy = (y - 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); //배열 3번째 
                int indexNxPy = (y + 1) * stride + x; // 배열 4번째
                int indexPxPy = (y + 1) * stride + (x + 1);

                //result x
                resultX += kernelX[0] * pixelData[indexMxMy];
                //resultX += kernelX[1] * pixelData[indexNxMy];
                resultX += kernelX[2] * pixelData[indexPxMy];

                resultX += kernelX[3] * pixelData[indexMxNy];
                //resultX += kernelX[4] * pixelData[indexNxNy];
                resultX += kernelX[5] * pixelData[indexPxNy];

                resultX += kernelX[6] * pixelData[indexMxPy];
                //resultX += kernelX[7] * pixelData[indexNxPy];
                resultX += kernelX[8] * pixelData[indexPxPy];

                // reuslt y
                resultY += kernelY[0] * pixelData[indexMxMy];
                resultY += kernelY[1] * pixelData[indexNxMy];
                resultY += kernelY[2] * pixelData[indexPxMy];
                //resultY += kernelY[3] * pixelData[indexMxNy];
                //resultY += kernelY[4] * pixelData[indexNxNy];
                //resultY += kernelY[5] * pixelData[indexPxNy];
                resultY += kernelY[6] * pixelData[indexMxPy];
                resultY += kernelY[7] * pixelData[indexNxPy];
                resultY += kernelY[8] * pixelData[indexPxPy];

                int sum = abs(resultX) + abs(resultY);

                // value validation
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                pixelData[indexNxNy] = static_cast<unsigned char>(sum);
            }
        }
    }
    // 라플라시안 완료
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernelTwo[9] = { 1, 1, 1, 
                          1, -8, 1,
                          1, 1, 1 };
        int kernel[9] = { 0, 1, 0,
                            1, -4, 1, 
                            0, 1, 0 };
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        for (int y = 0; y < height-1; ++y)
        {
            for (int x = 0; x < width-1; ++x)
            {
                double sum = 0.0;
                // 1번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); // 배열 0번째
                int indexNxMy = (y - 1) * stride + x; // 배열 1번째
                int indexPxMy = (y - 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); //배열 3번째 
                int indexNxPy = (y + 1) * stride + x; // 배열 4번째
                int indexPxPy = (y + 1) * stride + (x + 1);

                // vaild value
                sum += kernel[0] * pixelData[indexMxMy];
                sum += kernel[1] * pixelData[indexNxMy];
                sum += kernel[2] * pixelData[indexPxMy];

                sum += kernel[3] * pixelData[indexMxNy];
                sum += kernel[4] * pixelData[indexNxNy];
                sum += kernel[5] * pixelData[indexPxNy];

                sum += kernel[6] * pixelData[indexMxPy];
                sum += kernel[7] * pixelData[indexNxPy];
                sum += kernel[8] * pixelData[indexPxPy];

                sum /= 9;
                // vaild value
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                pixelData[indexNxNy] = static_cast<unsigned char>(sum);
               
            }
        }
    }

    // Blurring
    // 평균 블러 - 하는 중
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int kernel[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1  };
        for (int y = 1; y < height-1; ++y)
        {
            for (int x = 1; x < width-1; ++x)
            {
                double sum = 0.0;
                // 1번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); // 배열 0번째
                int indexNxMy = (y - 1) * stride + x; // 배열 1번째
                int indexPxMy = (y - 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); //배열 3번째 
                int indexNxPy = (y + 1) * stride + x; // 배열 4번째
                int indexPxPy = (y + 1) * stride + (x + 1);
                
                // vaild value
                sum += kernel[0] * pixelData[indexMxMy];
                sum += kernel[1] * pixelData[indexNxMy];
                sum += kernel[2] * pixelData[indexPxMy];

                sum += kernel[3] * pixelData[indexMxNy];
                sum += kernel[4] * pixelData[indexNxNy];
                sum += kernel[5] * pixelData[indexPxNy];

                sum += kernel[6] * pixelData[indexMxPy];
                sum += kernel[7] * pixelData[indexNxPy];
                sum += kernel[8] * pixelData[indexPxPy];
                
                sum /= 9;
                // vaild value
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                pixelData[indexNxNy] = static_cast<unsigned char>(sum);
            }
        }
    }
    // 가우시안 블러 - 시작안함
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int kernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double sum = 0.0;
                // 1번째 행
                int indexMxMy = (y - 1) * stride + (x - 1); // 배열 0번째
                int indexNxMy = (y - 1) * stride + x; // 배열 1번째
                int indexPxMy = (y - 1) * stride + (x + 1); // 배열 2번째
                // 2번째 행
                int indexMxNy = y * stride + (x - 1); //배열 3번째 
                int indexNxNy = y * stride + x; // 배열 4번째 중심
                int indexPxNy = y * stride + (x + 1);
                // 3번째 행
                int indexMxPy = (y + 1) * stride + (x - 1); //배열 3번째 
                int indexNxPy = (y + 1) * stride + x; // 배열 4번째
                int indexPxPy = (y + 1) * stride + (x + 1);

                // vaild value
                sum += kernel[0] * pixelData[indexMxMy];
                sum += kernel[1] * pixelData[indexNxMy];
                sum += kernel[2] * pixelData[indexPxMy];

                sum += kernel[3] * pixelData[indexMxNy];
                sum += kernel[4] * pixelData[indexNxNy];
                sum += kernel[5] * pixelData[indexPxNy];

                sum += kernel[6] * pixelData[indexMxPy];
                sum += kernel[7] * pixelData[indexNxPy];
                sum += kernel[8] * pixelData[indexPxPy];

                sum /= 16;
                // vaild value
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                pixelData[indexNxNy] = static_cast<unsigned char>(sum);
            }
        }
    }

    // Morphorogy
    // Dilation - 시작 안함
    void NativeCore::ApplyDilation(void* pixels, int width, int height, int stride, unsigned char threshold)
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
    // Erosion - 시작 안함
    void NativeCore::ApplyErosion(void* pixels, int width, int height, int stride, unsigned char threshold)
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
