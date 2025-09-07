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
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyEqualization(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        // Edge detect process
        public BitmapSource ApplyDifferential(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDifferential(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplySobel(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplySobel(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyLaplacian(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Blur process
        public BitmapSource ApplyAverageBlur(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyAverageBlur(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyGaussianBlur(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyGaussianBlur(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Mophorogy
        public BitmapSource ApplyDilation(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDilation(pixelPtr, width, height, stride, threshold);
            }, 128);

        }

        public BitmapSource ApplyErosion(BitmapSource source)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyErosion(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Image Matching
        public BitmapSource ApplyNCC(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride, threshold) =>
            {
                NativeProcessor.ApplyNCC(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, threshold);
            }, 128);
        }

        public BitmapSource ApplySAD(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride, threshold) =>
            {
                NativeProcessor.ApplySAD(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, threshold);
            }, 128);
        }

        public BitmapSource ApplySSD(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride, threshold) =>
            {
                NativeProcessor.ApplySSD(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, threshold);
            }, 128);
        }

    }

}