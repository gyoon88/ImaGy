using ImaGy.Wrapper;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public class MatchingProcessor
    {
        // Image Matching
        public BitmapSource ApplyNCC(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride, coordPtr) =>
            {
                NativeProcessor.ApplyNCC(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, coordPtr);
            });
        }

        public BitmapSource ApplySAD(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride, coordPtr) =>
            {
                NativeProcessor.ApplySAD(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, coordPtr);
            });
        }

        public BitmapSource ApplySSD(BitmapSource source, BitmapSource template)
        {
            return BitmapProcessorHelper.ProcessTwoBitmapSourcePixels(source, template, (sourcePixelPtr, sourceWidth,
                sourceHeight, sourceStride, templatePixelPtr, templateWidth, templateHeight, templateStride, coordPtr) =>
            {
                NativeProcessor.ApplySSD(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride, templatePixelPtr,
                    templateWidth, templateHeight, templateStride, coordPtr);
            });
        }
    }
}