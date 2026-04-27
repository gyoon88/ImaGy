using System.IO;
using System.Windows.Media.Imaging;

namespace ImaGy.Services;

public static class MatBitmapConverter
{
    public static BitmapSource FromPngBytes(byte[] png)
    {
        using var ms = new MemoryStream(png);
        var img = new BitmapImage();
        img.BeginInit();
        img.StreamSource = ms;
        img.CacheOption = BitmapCacheOption.OnLoad;
        img.EndInit();
        img.Freeze();
        return img;
    }
}
