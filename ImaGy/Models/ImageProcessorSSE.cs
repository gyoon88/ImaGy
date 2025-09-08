using ImaGy.Wrapper;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public class ImageProcessorSSE
    {
        // Edge detect process
        public BitmapSource ApplyDifferentialSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDifferentialSse(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

        public BitmapSource ApplySobelSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplySobelSse(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

        public BitmapSource ApplyLaplacianSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyLaplacianSse(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }


        // Blur process
        public BitmapSource ApplyAverageBlurSse(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyAverageBlurSse(pixelPtr, width, height, stride, kernelSize);
            });
        }

        public BitmapSource ApplyGaussianBlurSse(BitmapSource source, double sigma, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyGaussianBlurSse(pixelPtr, width, height, stride, sigma, kernelSize);
            });
        }


        // Mophorogy
        public BitmapSource ApplyDilationSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDilationSse(pixelPtr, width, height, stride, 128);
            });

        }

        public BitmapSource ApplyErosionSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyErosionSse(pixelPtr, width, height, stride, 128);
            });
        }


        //// Image Matching
        //public BitmapSource ApplyNCCSse(BitmapSource source)
        //{
        //    return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
        //    {
        //        NativeProcessor.ApplyNCCSse(pixelPtr, width, height, stride, threshold);
        //    }, 128);
        //}

        //public BitmapSource ApplySADSse(BitmapSource source)
        //{
        //    return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
        //    {
        //        NativeProcessor.ApplySADSse(pixelPtr, width, height, stride, threshold);
        //    }, 128);
        //}

        //public BitmapSource ApplySSDSse(BitmapSource source)
        //{
        //    return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
        //    {
        //        NativeProcessor.ApplySSDSse(pixelPtr, width, height, stride, threshold);
        //    }, 128);
        //}
    }
}