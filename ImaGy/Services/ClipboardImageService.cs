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
                Clipboard.SetImage(image);
            }
        }

        public BitmapSource? GetImage()
        {
            if (Clipboard.ContainsImage())
            {
                return Clipboard.GetImage();
            }
            return null;
        }
    }
}