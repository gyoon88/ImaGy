#pragma once

#include <vector> // std::vector�� ����ϹǷ� �����ؾ� �մϴ�.


namespace ImaGyNative
{
	static std::vector<double> createSobelKernelX(int kernelSize);
	static std::vector<double> createSobelKernelY(int kernelSize);

	static std::vector<double> createLaplacianKernel(int kernelSize);

	static std::vector<double> createGaussianKernel(int kernelSize, double sigma, bool isCircular);
	static std::vector<double> createAverageKernel(int kernelSize, bool isCircular);

	static int OtsuThreshold(const unsigned char* sourcePixels, int width, int height, int stride);

}
