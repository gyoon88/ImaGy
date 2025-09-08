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