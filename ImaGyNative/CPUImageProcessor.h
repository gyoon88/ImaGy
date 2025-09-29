#pragma once

#include <vector>
#include <complex>

namespace ImaGyNative
{

	// 필터 종류를 선택 enum
	enum class FilterType {
		LowPass,
		HighPass
	};
	void ApplyConvolution(const unsigned char* sourcePixels, unsigned char* destPixels,
		int width, int height, int stride, const std::vector<double>& kernel, int kernelSize);

	void ApplyConvolutionColor(const unsigned char* sourcePixels, unsigned char* destPixels,
		int width, int height, int stride, const std::vector<double>& kernel, int kernelSize);

	void ApplyBinarization_CPU(void* pixels, int width, int height, int stride, int threshold);
	void ApplyEqualization_CPU(void* pixels, int width, int height, int stride, unsigned char threshold);


	void ApplyDifferential_CPU(void* pixels, int width, int height, int stride, unsigned char threshold);
	void ApplySobel_CPU(void* pixels, int width, int height, int stride, int kernelSize);
	void ApplyLaplacian_CPU(void* pixels, int width, int height, int stride, int kernelSize);

	void ApplyGaussianBlur_CPU(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);
	void ApplyAverageBlur_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

	void ApplyDilation_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
	void ApplyErosion_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

	void ApplyNCC_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight,
		int templateStride, int* outCoords);
	void ApplySAD_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);
	void ApplySSD_CPU(void* pixels, int width, int height, int stride, void* templatePixels, int templateWidth, int templateHeight, int templateStride, int* outCoords);

	/////
	// Color
	// CPU Processing
	// 
	// Blur
	void ApplyAverageBlurColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
	void ApplyGaussianBlurColor_CPU(void* pixels, int width, int height, int stride, double sigma, int kernelSize, bool useCircularKernel);

	//Morphology
	void ApplyDilationColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);
	void ApplyErosionColor_CPU(void* pixels, int width, int height, int stride, int kernelSize, bool useCircularKernel);

	// Gray sclae 로 
	void ApplyFFT2DSpectrum_CPU(void* inputPixels, Complex* outputSpectrum, int width, int height, int stride, bool isInverse);
	void ApplyFFT2DPhase_CPU(void* pixels, Complex* outputSpectrum, int width, int height, int stride, bool isInverse);

	void ApplyFrequencyFilter_CPU(void* pixels, int width, int height, int stride, FilterType filterType, double radius);
	void ApplyAxialBandStopFilter_CPU(void* pixels, int width, int height, int stride, double lowFreqRadius, double bandThickness);

	// Clustering 
	void ApplyKMeansClustering_CPU(void* pixels, int width, int height, int stride, int k, int iteration);
	void ApplyKMeansClusteringXY_Normalized_CPU(void* pixels, int width, int height, int stride, int k, int iteration);
	void ApplyGmmSegmentation_CPU(void* pixels, int width, int height, int stride, int numClusters);

}