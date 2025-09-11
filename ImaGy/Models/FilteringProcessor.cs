using System.Windows.Media.Imaging;
using ImaGy.Wrapper;


namespace ImaGy.Models
{
    public class FilterProcessor
    {
        // Edge detect process
        public BitmapSource ApplyDifferential(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDifferential(pixelPtr, width, height, stride, 128);
            });
        }

        public BitmapSource ApplySobel(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixelsWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplySobel(pixelPtr, width, height, stride, kernelSize);
            });
        }

        public BitmapSource ApplyLaplacian(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixelsWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, kernelSize);
            });
        }
        public BitmapSource ApplyFFT(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessGrayscaleImageWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyFFT(pixelPtr, width, height, stride, kernelSize); 
            });
        }


        // Blur process
        public BitmapSource ApplyAverageBlur(BitmapSource source, int kernelSize, bool useCircularKernel)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixelsWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyAverageBlur(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
            });
        }

        public BitmapSource ApplyGaussianBlur(BitmapSource source, double sigma, int kernelSize, bool useCircularKernel)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixelsWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyGaussianBlur(pixelPtr, width, height, stride, sigma, kernelSize, useCircularKernel);
            });
        }
    }
}