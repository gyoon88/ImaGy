using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace ImaGy.Models
{
    public class ImageProcessor
    {
        // Colour | contrast
        public BitmapSource ApplyBinarization(BitmapSource source, byte threshold)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, threshold);
            });
        }

        public BitmapSource ApplyEqualization(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, 128); // Keep 128 as it's a fixed value for equalization
            });
        }

        // Edge detect process
        public BitmapSource ApplyDifferential(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDifferential(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

        public BitmapSource ApplySobel(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplySobel(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

        public BitmapSource ApplyLaplacian(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
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


        // Mophorogy
        public BitmapSource ApplyDilation(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDilation(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });

        }

        public BitmapSource ApplyErosion(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyErosion(pixelPtr, width, height, stride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }


        // Image Matching
        public BitmapSource ApplyNCC(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride) =>
            {
                NativeProcessor.ApplyNCC(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

        public BitmapSource ApplySAD(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride) =>
            {
                NativeProcessor.ApplySAD(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

        public BitmapSource ApplySSD(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride) =>
            {
                NativeProcessor.ApplySSD(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, 128); // Assuming 128 is a default or placeholder threshold
            });
        }

    }

}