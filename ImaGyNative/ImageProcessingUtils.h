#pragma once

#include <vector>

namespace ImaGyNative
{
	std::vector<double> createSobelKernelX(int kernelSize);
	std::vector<double> createSobelKernelY(int kernelSize);
	std::vector<double> createLaplacianKernel(int kernelSize);
	std::vector<double> createGaussianKernel(int kernelSize, double sigma, bool isCircular);
	std::vector<double> createAverageKernel(int kernelSize, bool isCircular);
	int OtsuThreshold(const unsigned char* sourcePixels, int width, int height, int stride);

    struct Complex {
        double real;
        double imag;
    };

    // 덧셈
    inline Complex operator+(const Complex& a, const Complex& b) {
        return { a.real + b.real, a.imag + b.imag };
    }

    // 뺄셈
    inline Complex operator-(const Complex& a, const Complex& b) {
        return { a.real - b.real, a.imag - b.imag };
    }

    // 곱셈 (Complex * Complex)
    inline Complex operator*(const Complex& a, const Complex& b) {
        return {
            a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real
        };
    }

    inline void swap(Complex& a, Complex& b) noexcept
    {
        // 내부 멤버들은 기본 타입(double)이므로 std::swap을 그대로 사용합니다.
        std::swap(a.real, b.real);
        std::swap(a.imag, b.imag);
    }
    /// <summary>
    /// Complex 타입을 스칼라(실수) 값으로 나누는 연산자입니다.
    /// </summary>
    inline Complex operator/(const Complex& a, double scalar) {
        if (scalar == 0) return { 0, 0 }; // 0으로 나누는 경우 예외 처리
        return { a.real / scalar, a.imag / scalar };
    }

    /// <summary>
    /// /= 복합 대입 연산자입니다.
    /// </summary>
    inline void operator/=(Complex& a, double scalar) {
        if (scalar == 0) {
            a.real = 0;
            a.imag = 0;
        }
        else {
            a.real /= scalar;
            a.imag /= scalar;
        }
    }

    // 곱셈 (Complex * double) - IFFT 스케일링에 필요
    inline Complex operator*(const Complex& a, double scalar) {
        return { a.real * scalar, a.imag * scalar };
    }
    void FFT_1D_Iterative(Complex* data, int N, bool isInverse);

    void FFT_1D_Recursive(Complex* data, int N, bool isInverse);
    void ApplyFFT2D_CPU(const void* inputPixels, Complex* outputSpectrum, int width, int height, int stride, bool isInverse);
    void FFT_Shift2D(Complex* spectrum, int width, int height);


    // Clustering

}
