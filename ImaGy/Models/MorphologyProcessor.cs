using ImaGy.Wrapper;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public class MorphologyProcessor
    {
        // Mophorogy
        public BitmapSource ApplyDilation(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixelsWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyDilation(pixelPtr, width, height, stride, kernelSize); 
            });

        }

        public BitmapSource ApplyErosion(BitmapSource source, int kernelSize)
        {
            return BitmapProcessorHelper.ProcessBitmapSourcePixelsWithPadding(source, kernelSize, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyErosion(pixelPtr, width, height, stride, kernelSize);
            });
        }
    }
}