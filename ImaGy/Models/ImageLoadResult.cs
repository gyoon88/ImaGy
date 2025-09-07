using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public class ImageLoadResult
    {
        public BitmapSource? Bitmap { get; set; }
        public string? FileName { get; set; }
        public string? Resolution { get; set; }
        public double LoadTime { get; set; }
    }
}