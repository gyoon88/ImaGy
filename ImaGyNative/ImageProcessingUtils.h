#pragma once

#include <vector> // std::vector ϹǷ ؾ մϴ.

namespace ImaGyNative
{
	std::vector<double> createSobelKernelX(int kernelSize);
	std::vector<double> createSobelKernelY(int kernelSize);

	std::vector<double> createLaplacianKernel(int kernelSize);

	std::vector<double> createGaussianKernel(int kernelSize, double sigma, bool isCircular);
	std::vector<double> createAverageKernel(int kernelSize, bool isCircular);

	int OtsuThreshold(const unsigned char* sourcePixels, int width, int height, int stride);
}