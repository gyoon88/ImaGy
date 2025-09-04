using ImaGy.Wrapper;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Model
{
    internal class ImageProcessorSSE
    {
        // Edge detect process
        public BitmapSource ApplyDifferentialSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDifferentialSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplySobelSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplySobelSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyLaplacianSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyLaplacianSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Blur process
        public BitmapSource ApplyAverageBlurSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyAverageBlurSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyGaussianBlurSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyGaussianBlurSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Mophorogy
        public BitmapSource ApplyDilationSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDilationSse(pixelPtr, width, height, stride, threshold);
            }, 128);

        }

        public BitmapSource ApplyErosionSse(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyErosionSse(pixelPtr, width, height, stride, threshold);
            }, 128);
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