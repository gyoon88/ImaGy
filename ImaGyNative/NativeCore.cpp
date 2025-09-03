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

                // �̿� �ȼ� �ε��� (�� �κ��� �� ������ �� ������, �������� ���� �״�� �Ӵϴ�)
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // ������� ����
                for (int i = 0; i < 9; ++i)
                {
                    sum += kernel[i] * sourcePixels[indexes[i]];
                }
                           
                if (kernelSum != 0) {  // 0���� ������ ���� ����
                    sum /= kernelSum;
                }

                // ��� �� ���� ���� (Clamping)
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                destPixels[indexes[4]] = static_cast<unsigned char>(sum);
            }
        }
    }
    // // Color Contrast
    // ����ȭ - �Ϸ�
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
    // ��Ȱȭ - ���۾���.
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
    // �̺� �Ϸ�
    void NativeCore::ApplyDifferential(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // 1. ����� ������ ���ο� ���۸� �����մϴ�.
        unsigned char* resultBuffer = new unsigned char[height * stride];

        for (int y = 0; y < height - 1; ++y)
        {
            for (int x = 0; x < width - 1; ++x)
            {
                int centerIndex = y * stride + x;
                int indexPx = y * stride + (x + 1); // ������ �ȼ�
                int indexPy = (y + 1) * stride + x; // �Ʒ��� �ȼ�

                // 2. �������� ���� ����� �����ϴ�.
                int gradX = pixelData[indexPx] - pixelData[centerIndex]; // ���� ��ȭ��
                int gradY = pixelData[indexPy] - pixelData[centerIndex]; // ���� ��ȭ��

                int val = abs(gradX) + abs(gradY);

                // 3. (����) threshold �Ķ���͸� ����մϴ�.
                // ��ȭ���� threshold���� ũ�� ����(���), �ƴϸ� ���(������)���� ó���մϴ�.
                unsigned char finalValue = (val > threshold) ? 255 : 0;

                // 4. ����� �� ���ۿ� �����մϴ�.
                resultBuffer[centerIndex] = finalValue;
            }
        }

        // 5. ��� ����� ���� ��, ��� ������ ������ ���� ���۷� �����մϴ�.
        memcpy(pixelData, resultBuffer, height * stride);

        // 6. �Ҵ��ߴ� �޸𸮸� �����մϴ�.
        delete[] resultBuffer;
    }
    // �Һ� �Ϸ�
    void NativeCore::ApplySobel(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        // 1. Gx, Gy Ŀ�� ����
        int kernelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        int kernelY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);

        // ���� �����͸� �����ص� ���� ���� (�б� �������� ���)
        unsigned char* sourceBuffer = new unsigned char[height * stride];
        memcpy(sourceBuffer, pixelData, height * stride);

        // Gx, Gy ���� ����� ������ �ӽ� ���� 2�� ����
        double* bufferX = new double[height * stride];
        double* bufferY = new double[height * stride];

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                double sumX = 0.0;
                double sumY = 0.0;

                // --- ä������ �κ� ���� ---

                // 9�� �̿� �ȼ��� �ε��� ���
                int indexes[9] = {
                    (y - 1) * stride + (x - 1), (y - 1) * stride + x, (y - 1) * stride + (x + 1),
                    y * stride + (x - 1),       y * stride + x,       y * stride + (x + 1),
                    (y + 1) * stride + (x - 1), (y + 1) * stride + x, (y + 1) * stride + (x + 1)
                };

                // Gx, Gy ���� ������� ����
                for (int i = 0; i < 9; ++i)
                {
                    // ���� ������(sourceBuffer)���� ���� �о�;� �մϴ�.
                    sumX += kernelX[i] * sourceBuffer[indexes[i]];
                    sumY += kernelY[i] * sourceBuffer[indexes[i]];
                }

                // --- ä������ �κ� �� ---

                int centerIndex = y * stride + x;
                bufferX[centerIndex] = sumX;
                bufferY[centerIndex] = sumY;
            }
        }

        // 3. �� ������ ����� ���ļ� ���� �ȼ� �� ���
        for (int i = 0; i < height * stride; ++i)
        {
            double finalValue = sqrt(bufferX[i] * bufferX[i] + bufferY[i] * bufferY[i]); // �� �� ��Ȯ�� Gx, Gy ���� ���
            // double finalValue = abs(bufferX[i]) + abs(bufferY[i]); // �� ���� �ٻ�ġ ���
            if (finalValue > 255) finalValue = 255;
            pixelData[i] = static_cast<unsigned char>(finalValue);
        }

        // 4. �Ҵ�� �޸� ����
        delete[] sourceBuffer;
        delete[] bufferX;
        delete[] bufferY;
    }
    // ���ö�þ� �Ϸ�
    void NativeCore::ApplyLaplacian(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 1, 1, 1, -8, 1, 1, 1, 1 };

        unsigned char* pixelData = static_cast<unsigned char*>(pixels); // ���� ����
        unsigned char* resultBuffer = new unsigned char[height * stride]; // ������ ���� ũ���� ��� ���� ����

        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, 0); // ������� ���� �Լ� ȣ��
        memcpy(pixelData, resultBuffer, height * stride); //����� ���� ���ۿ� ����

        delete[] resultBuffer; //�޸� ����
    }

    // Blurring
    void NativeCore::ApplyAverageBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        double kernelSum = 9.0;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // ���� ������� �Լ� ȣ��
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, kernelSum);

        // ���� ����� ���� ���ۿ� ����
        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
    }
    // ����þ� �� 
    void NativeCore::ApplyGaussianBlur(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        int kernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
        double kernelSum = 16.0;

        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        unsigned char* resultBuffer = new unsigned char[height * stride];

        // ���� ������� �Լ� ȣ��
        ApplyConvolution3x3(pixelData, resultBuffer, width, height, stride, kernel, kernelSum);

        // ���� ����� ���� ���ۿ� ����
        memcpy(pixelData, resultBuffer, height * stride);
        delete[] resultBuffer;
    }

    // Morphorogy
    // Dilation - ���� ����
    void NativeCore::ApplyDilation(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int arr[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 ,1 / 9 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // 1��° ��
                int indexMxPy = (y + 1) * stride + (x - 1); // �迭 0��°
                int indexNxPy = (y + 1) * stride + x; // �迭 1��°
                int indexPxPy = (y + 1) * stride + (x + 1); // �迭 2��°
                // 2��° ��
                int indexMxNy = y * stride + (x - 1); //�迭 3��° 
                int indexNxNy = y * stride + x; // �迭 4��° �߽�
                int indexPxNy = y * stride + (x + 1);
                // 3��° ��
                int indexMxMy = (y - 1) * stride + (x - 1); //�迭 3��° 
                int indexNxMy = (y - 1) * stride + x; // �迭 4��°
                int indexPxMy = (y - 1) * stride + (x + 1);



                pixelData[indexNxNy];
            }
        }
    }
    // Erosion - ���� ����
    void NativeCore::ApplyErosion(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int arr[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 ,1 / 9 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // 1��° ��
                int indexMxPy = (y + 1) * stride + (x - 1); // �迭 0��°
                int indexNxPy = (y + 1) * stride + x; // �迭 1��°
                int indexPxPy = (y + 1) * stride + (x + 1); // �迭 2��°
                // 2��° ��
                int indexMxNy = y * stride + (x - 1); //�迭 3��° 
                int indexNxNy = y * stride + x; // �迭 4��° �߽�
                int indexPxNy = y * stride + (x + 1);
                // 3��° ��
                int indexMxMy = (y - 1) * stride + (x - 1); //�迭 3��° 
                int indexNxMy = (y - 1) * stride + x; // �迭 4��°
                int indexPxMy = (y - 1) * stride + (x + 1);



                pixelData[indexNxNy];
            }
        }
    }

    // Image Matching - ���� �̹��ֿ� ���� ��
    // normailized cross correlation
    void NativeCore::ApplyNCC(void* pixels, int width, int height, int stride, unsigned char threshold)
    {
        unsigned char* pixelData = static_cast<unsigned char*>(pixels);
        int arr[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 ,1 / 9 };
        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                // 1��° ��
                int indexMxPy = (y + 1) * stride + (x - 1); // �迭 0��°
                int indexNxPy = (y + 1) * stride + x; // �迭 1��°
                int indexPxPy = (y + 1) * stride + (x + 1); // �迭 2��°
                // 2��° ��
                int indexMxNy = y * stride + (x - 1); //�迭 3��° 
                int indexNxNy = y * stride + x; // �迭 4��° �߽�
                int indexPxNy = y * stride + (x + 1);
                // 3��° ��
                int indexMxMy = (y - 1) * stride + (x - 1); //�迭 3��° 
                int indexNxMy = (y - 1) * stride + x; // �迭 4��°
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
                // 1��° ��
                int indexMxPy = (y + 1) * stride + (x - 1); // �迭 0��°
                int indexNxPy = (y + 1) * stride + x; // �迭 1��°
                int indexPxPy = (y + 1) * stride + (x + 1); // �迭 2��°
                // 2��° ��
                int indexMxNy = y * stride + (x - 1); //�迭 3��° 
                int indexNxNy = y * stride + x; // �迭 4��° �߽�
                int indexPxNy = y * stride + (x + 1);
                // 3��° ��
                int indexMxMy = (y - 1) * stride + (x - 1); //�迭 3��° 
                int indexNxMy = (y - 1) * stride + x; // �迭 4��°
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
                // 1��° ��
                int indexMxPy = (y + 1) * stride + (x - 1); // �迭 0��°
                int indexNxPy = (y + 1) * stride + x; // �迭 1��°
                int indexPxPy = (y + 1) * stride + (x + 1); // �迭 2��°
                // 2��° ��
                int indexMxNy = y * stride + (x - 1); //�迭 3��° 
                int indexNxNy = y * stride + x; // �迭 4��° �߽�
                int indexPxNy = y * stride + (x + 1);
                // 3��° ��
                int indexMxMy = (y - 1) * stride + (x - 1); //�迭 3��° 
                int indexNxMy = (y - 1) * stride + x; // �迭 4��°
                int indexPxMy = (y - 1) * stride + (x + 1);



                pixelData[indexNxNy];
            }
        }
    }

}
