using System.Windows;
using System.Windows.Media.Imaging;

namespace ImaGy.Services
{
    public class ClipboardImageService
    {
        public void SetImage(BitmapSource image)
        {
            if (image != null)
            {
                System.Windows.Clipboard.SetImage(image);
            }
        }

        public BitmapSource? GetImage()
        {
            if (System.Windows.Clipboard.ContainsImage())
            {
                return System.Windows.Clipboard.GetImage();
            }
            return null;
        }
    }
}