using ImaGy.Wrapper;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public class MorphologyProcessor
    {
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
    }
}