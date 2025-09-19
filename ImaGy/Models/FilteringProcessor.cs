using ImaGy.Wrapper;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;


namespace ImaGy.Models
{
    public class FilterProcessor
    {
        // Edge detect process
        public BitmapSource ApplyDifferential(BitmapSource source, bool isColor)
        {
            BitmapSource gray;
            if (isColor)
            {
                source = new FormatConvertedBitmap(source, PixelFormats.Gray8, null, 0);
            }
            return BitmapProcessorHelper.ApplyEffect(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDifferential(pixelPtr, width, height, stride, 128);
            });
        }

        public BitmapSource ApplySobel(BitmapSource source, int kernelSize, bool isColor)

        {
            BitmapSource gray;
            if (isColor)
            {
                source = new FormatConvertedBitmap(source, PixelFormats.Gray8, null, 0);
            }
            return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplySobel(pixelPtr, width, height, stride, kernelSize);
            });
        }

        public BitmapSource ApplyLaplacian(BitmapSource source, int kernelSize, bool isColor)
        {
            BitmapSource gray;
            if (isColor)
            {
                source = new FormatConvertedBitmap(source, PixelFormats.Gray8, null, 0);
            }
            

            return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, kernelSize);
            });
        }
        public BitmapSource ApplyFFT(BitmapSource source, int kernelSize, bool isColor, bool isInverse, bool isCPU, bool isPhase)
        {
            return BitmapProcessorHelper.ApplyFFTEffect(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyFFT(pixelPtr, width, height, stride, kernelSize, isInverse, isCPU, isPhase);
            });
            

        }

        // Blur process
        public BitmapSource ApplyAverageBlur(BitmapSource source, int kernelSize, bool useCircularKernel, bool isColor)
        {
            if (isColor)
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyAverageBlurColor(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
                });
            }
            else
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyAverageBlur(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
                });
            }
        }

        public BitmapSource ApplyGaussianBlur(BitmapSource source, double sigma, int kernelSize, bool useCircularKernel, bool isColor)
        {
            if(isColor){
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyGaussianBlurColor(pixelPtr, width, height, stride, sigma, kernelSize, useCircularKernel);
                });
            }
            else
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyGaussianBlur(pixelPtr, width, height, stride, sigma, kernelSize, useCircularKernel);
                });
            }
        }
    }
}