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
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplySobel(pixelPtr, width, height, stride, kernelSize);
            });
        }

        public BitmapSource ApplyLaplacian(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, kernelSize);
            });
        }
        public BitmapSource ApplyFFT(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, 128); 
            });
        }


        // Blur process
        public BitmapSource ApplyAverageBlur(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyAverageBlur(pixelPtr, width, height, stride, kernelSize);
            });
        }

        public BitmapSource ApplyGaussianBlur(BitmapSource source, double sigma, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyGaussianBlur(pixelPtr, width, height, stride, sigma, kernelSize);
            });
        }
    }
}