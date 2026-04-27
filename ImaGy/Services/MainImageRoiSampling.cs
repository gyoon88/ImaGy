using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ImaGy.Grids;

namespace ImaGy.Services;

/// <summary>
/// 메인 뷰어 BitmapSource + ROI 에서 그레이(명암) 샘플·행/열 합·FloatGrid 변환.
/// </summary>
public static class MainImageRoiSampling
{
    public static Int32Rect ClipToBitmap(Int32Rect r, int pixelWidth, int pixelHeight)
    {
        int x = Math.Clamp(r.X, 0, Math.Max(0, pixelWidth - 1));
        int y = Math.Clamp(r.Y, 0, Math.Max(0, pixelHeight - 1));
        int w = Math.Clamp(r.Width, 1, pixelWidth - x);
        int h = Math.Clamp(r.Height, 1, pixelHeight - y);
        return new Int32Rect(x, y, w, h);
    }

    /// <summary>이미지 레이아웃(DIP) 한 칸이 비트맵 픽셀 몇 칸에 해당하는지. LayoutTransform·Stretch 반영 후 <see cref="Image.ActualWidth"/> 기준.</summary>
    public static bool TryGetDipPerPixel(System.Windows.Controls.Image? img, out double dipPerPixelX, out double dipPerPixelY)
    {
        dipPerPixelX = dipPerPixelY = 1;
        if (img?.Source is not BitmapSource bmp || bmp.PixelWidth < 1 || bmp.PixelHeight < 1)
            return false;
        if (img.ActualWidth < 1e-6 || img.ActualHeight < 1e-6)
            return false;
        dipPerPixelX = img.ActualWidth / bmp.PixelWidth;
        dipPerPixelY = img.ActualHeight / bmp.PixelHeight;
        return true;
    }

    /// <summary>뷰어 상의 드래그 사각형(이미지 좌표, DIP)을 픽셀 인덱스로 변환.</summary>
    public static Int32Rect ViewRectToPixelRect(Rect viewRect, System.Windows.Controls.Image img, int pixelWidth, int pixelHeight)
    {
        if (!TryGetDipPerPixel(img, out var dppx, out var dppy) || dppx < 1e-12 || dppy < 1e-12)
        {
            dppx = dppy = 1;
        }

        double x0 = Math.Min(viewRect.Left, viewRect.Right) / dppx;
        double y0 = Math.Min(viewRect.Top, viewRect.Bottom) / dppy;
        double x1 = Math.Max(viewRect.Left, viewRect.Right) / dppx;
        double y1 = Math.Max(viewRect.Top, viewRect.Bottom) / dppy;
        int ix0 = (int)Math.Floor(x0);
        int iy0 = (int)Math.Floor(y0);
        int ix1 = (int)Math.Ceiling(x1);
        int iy1 = (int)Math.Ceiling(y1);
        int rw = Math.Max(1, ix1 - ix0);
        int rh = Math.Max(1, iy1 - iy0);
        return ClipToBitmap(new Int32Rect(ix0, iy0, rw, rh), pixelWidth, pixelHeight);
    }

    /// <summary>ROI 또는 전체를 명암 Gray8 한 장으로.</summary>
    public static BitmapSource ToGray8Cropped(BitmapSource source, Int32Rect? roi)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        BitmapSource work = source;
        if (roi.HasValue)
        {
            var r = ClipToBitmap(roi.Value, source.PixelWidth, source.PixelHeight);
            work = new CroppedBitmap(source, r);
        }

        if (work.Format == PixelFormats.Gray8)
            return work;

        var gray = new FormatConvertedBitmap();
        gray.BeginInit();
        gray.Source = work;
        gray.DestinationFormat = PixelFormats.Gray8;
        gray.EndInit();
        return gray;
    }

    /// <summary>각 행마다 ROI 가로 구간의 밝기 합 (길이 = ROI 높이).</summary>
    public static double[] RowSumsAlongX(BitmapSource gray8CroppedOrFull)
    {
        int w = gray8CroppedOrFull.PixelWidth;
        int h = gray8CroppedOrFull.PixelHeight;
        int stride = w;
        var buf = new byte[stride * h];
        gray8CroppedOrFull.CopyPixels(buf, stride, 0);
        var sums = new double[h];
        for (int y = 0; y < h; y++)
        {
            long s = 0;
            int row = y * stride;
            for (int x = 0; x < w; x++)
                s += buf[row + x];
            sums[y] = s;
        }

        return sums;
    }

    /// <summary>각 열마다 ROI 세로 구간의 밝기 합 (길이 = ROI 너비).</summary>
    public static double[] ColSumsAlongY(BitmapSource gray8CroppedOrFull)
    {
        int w = gray8CroppedOrFull.PixelWidth;
        int h = gray8CroppedOrFull.PixelHeight;
        int stride = w;
        var buf = new byte[stride * h];
        gray8CroppedOrFull.CopyPixels(buf, stride, 0);
        var sums = new double[w];
        for (int x = 0; x < w; x++)
        {
            long s = 0;
            for (int y = 0; y < h; y++)
                s += buf[y * stride + x];
            sums[x] = s;
        }

        return sums;
    }

    /// <summary>라인 프로파일용: ROI(또는 전체) 명암을 FloatGrid 로.</summary>
    public static FloatGrid ToLuminanceFloatGrid(BitmapSource source, Int32Rect? roi)
    {
        var gray = ToGray8Cropped(source, roi);
        int w = gray.PixelWidth;
        int h = gray.PixelHeight;
        int stride = w;
        var buf = new byte[stride * h];
        gray.CopyPixels(buf, stride, 0);
        var data = new double[w * h];
        for (int i = 0; i < data.Length; i++)
            data[i] = buf[i];
        return new FloatGrid(h, w, data);
    }
}
