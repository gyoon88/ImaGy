// NativeCore.cpp
#include "pch.h"
#include "NativeCore.h"
#include <cmath>

namespace ImaGyNative
{
    void ApplyConvolution3x3( const unsigned char* sourcePixels, unsigned char* destPixels, 
        int width, int height, int stride, const int kernel[9], double kernelSum)
    {
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double sum = 0.0;

                // 이웃 픽셀 인덱스 (이 부분은 더 압축할 수 있지만, 가독성을 위해 그대로 둡니다)
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // 컨볼루션 연산
                for (int i = 0; i < 9; ++i)
                {
                    sum += kernel[i] * sourcePixels[indexes[i]];
                }
                           
                if (kernelSum != 0) {  // 0으로 나누는 것을 방지
                    sum /= kernelSum;
                }

                // 결과 값 범위 조정 (Clamping)
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                destPixels[indexes[4]] = static_cast<unsigned char>(sum);
            }
        }
    }
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
    // 미분 완료
    void NativeCore::ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // 1. 결과를 저장할 새로운 버퍼를 생성합니다.
        unsigned char* resultBuffer = new unsigned char[height * stride];

        for (int y = 0; y < height - 1; ++y)
        {
            for (int x = 0; x < width - 1; ++x)
            {
                int centerIndex = y * stride + x;
                int indexPx = y * stride + (x + 1); // 오른쪽 픽셀
                int indexPy = (y + 1) * stride + x; // 아래쪽 픽셀

                // 2. 가독성을 위해 계산을 나눕니다.
                int gradX = pixelData[indexPx] - pixelData[centerIndex]; // 수평 변화량
                int gradY = pixelData[indexPy] - pixelData[centerIndex]; // 수직 변화량

                int val = abs(gradX) + abs(gradY);

                // 3. (개선) threshold 파라미터를 사용합니다.
                // 변화량이 threshold보다 크면 엣지(흰색), 아니면 배경(검은색)으로 처리합니다.
                unsigned char finalValue = (val > threshold) ? 255 : 0;

                // 4. 결과를 새 버퍼에 저장합니다.
                resultBuffer[centerIndex] = finalValue;
            }
        }

        // 5. 모든 계산이 끝난 후, 결과 버퍼의 내용을 원본 버퍼로 복사합니다.
        memcpy(pixelData, resultBuffer, height * stride);

        // 6. 할당했던 메모리를 해제합니다.
        delete[] resultBuffer;
    }
    // 소벨 완료
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        // 1. Gx, Gy 커널 정의
        int kernelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        int kernelY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // 원본 데이터를 복사해둘 버퍼 생성 (읽기 전용으로 사용)
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        // Gx, Gy 연산 결과를 저장할 임시 버퍼 2개 생성
        double* bufferX = new double[height * stride];
        double* bufferY = new double[height * stride];

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double sumX = 0.0;
                double sumY = 0.0;

                // --- 채워넣은 부분 시작 ---

                // 9개 이웃 픽셀의 인덱스 계산
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // Gx, Gy 각각 컨볼루션 수행
                for (int i = 0; i < 9; ++i)
                {
                    // 원본 데이터(sourceBuffer)에서 값을 읽어와야 합니다.
                    sumX += kernelX[i] * sourceBuffer[indexes[i]];
                    sumY += kernelY[i] * sourceBuffer[indexes[i]];
                }

                // --- 채워넣은 부분 끝 ---

                int centerIndex = y * stride + x;
                bufferX[centerIndex] = sumX;
                bufferY[centerIndex] = sumY;
            }
        }

        // 3. 두 버퍼의 결과를 합쳐서 최종 픽셀 값 계산
        for (int i = 0; i < height * stride; ++i)
        {
            double finalValue = sqrt(bufferX[i] * bufferX[i] + bufferY[i] * bufferY[i]); // 좀 더 정확한 Gx, Gy 조합 방식
            // double finalValue = abs(bufferX[i]) + abs(bufferY[i]); // 더 빠른 근사치 방식
            if (finalValue > 255) finalValue = 255;
            pixelData[i] = static_cast<unsigned char>(finalValue);
        }

        // 4. 할당된 메모리 해제
        delete[] sourceBuffer;
        delete[] bufferX;
        delete[] bufferY;
    }
    // 라플라시안 완료
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 1, 1, 1, -8, 1, 1, 1, 1 };

        unsigned char* pixelData = static_cast<unsigned char*>(pixels); // 원본 버퍼
        unsigned char* resultBuffer = new unsigned char[height * stride]; // 원본과 같은 크기의 결과 버퍼 생성

        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, 0); // 컨볼루션 헬퍼 함수 호출
        memcpy(pixelData, resultBuffer, height * stride); //결과를 원본 버퍼에 복사

        delete[] resultBuffer; //메모리 해제
    }

    // Blurring
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        double kernelSum = 9.0;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // 범용 컨볼루션 함수 호출
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, kernelSum);

        // 최종 결과를 원본 버퍼에 복사
        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
    }
    // 가우시안 블러 
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
        double kernelSum = 16.0;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // 범용 컨볼루션 함수 호출
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, kernelSum);

        // 최종 결과를 원본 버퍼에 복사
        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
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
