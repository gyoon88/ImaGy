#pragma once
#include <vector> // std::vector  

namespace ImaGyNative
{
	bool LaunchBinarizationKernel(unsigned char* pixels, int width, int height, int stride, int threshold);
	bool LaunchEqualizationKernel(unsigned char* pixels, int width, int height, int stride);

	// Filters 
	bool LaunchGaussianBlurKernel(unsigned char* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
	bool LaunchAverageBlurKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
	bool LaunchSobelKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);
	bool LaunchLaplacianKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);

	// Morphology 
	bool LaunchDilationKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
	bool LaunchErosionKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

	// Template Matching
	bool LaunchNccKernel(const unsigned char* image, int width, int height, int stride,
		const unsigned char* templ, int tempWidth, int tempHeight, int tempStride,
		int* out_x, int* out_y);
	bool LaunchSadKernel(const unsigned char* image, int width, int height, int stride,
		const unsigned char* templ, int tempWidth, int tempHeight, int tempStride,
		int* out_x, int* out_y);
	bool LaunchSsdKernel(const unsigned char* image, int width, int height, int stride,
		const unsigned char* templ, int tempWidth, int tempHeight, int tempStride,
		int* out_x, int* out_y);

	// FFT Filter
	bool LaunchFftFilterKernel(unsigned char* pixels, int width, int height, int stride, int kernelSize);
	bool LaunchFftSpectrumKernel(unsigned char* pixels, int width, int height, int stride);
	bool LaunchKMeansKernel(void* pixels, int width, int height, int stride, int k, int iteration);
}