using ImaGy.Models;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Windows;

namespace ImaGy.Services
{
    public class CropService
    {
        public BitmapSource? CropImage(BitmapSource source, RoiModel roi)
        {
            if (source == null || roi == null) return null;

            int x = (int)roi.X;
            int y = (int)roi.Y;
            int width = (int)roi.Width;
            int height = (int)roi.Height;

            x = Math.Max(0, x);
            y = Math.Max(0, y);
            width = Math.Min(width, source.PixelWidth - x);
            height = Math.Min(height, source.PixelHeight - y);

            if (width <= 0 || height <= 0) return null; 

            CroppedBitmap croppedBitmap = new CroppedBitmap(
                source,
                new Int32Rect(x, y, width, height));

            croppedBitmap.Freeze(); // Freeze 로 다른 쓰레드에서 접근 가능! 
            return croppedBitmap;
        }
    }
}