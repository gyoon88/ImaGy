using System;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Services
{
    /// <summary>
    /// 밝기(가산), 대비(128 기준 선형), 감마(정규화 후 거듭제곱)를 Pbgra32 버퍼에 적용합니다.
    /// </summary>
    public static class ImageToneAdjustService
    {
        /// <summary>After 이미지를 편집용 Pbgra32 픽셀 버퍼로 복사합니다.</summary>
        public static (byte[] Pixels, int Width, int Height, int Stride, double DpiX, double DpiY, bool WasGray8) CapturePbgra32(BitmapSource source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));

            var conv = new FormatConvertedBitmap(source, PixelFormats.Pbgra32, null, 0);
            int w = conv.PixelWidth;
            int h = conv.PixelHeight;
            int stride = (w * PixelFormats.Pbgra32.BitsPerPixel + 7) / 8;
            var pixels = new byte[stride * h];
            conv.CopyPixels(pixels, stride, 0);
            bool wasGray = source.Format == PixelFormats.Gray8;
            return (pixels, w, h, stride, conv.DpiX, conv.DpiY, wasGray);
        }

        public static byte[] TransformPbgra(byte[] srcPixels, int stride, int height, double brightness, double contrast, double gamma)
        {
            contrast = Math.Clamp(contrast, 0.01, 10.0);
            gamma = Math.Clamp(gamma, 0.05, 10.0);
            brightness = Math.Clamp(brightness, -256.0, 256.0);

            var dst = new byte[srcPixels.Length];
            bool skipGamma = Math.Abs(gamma - 1.0) < 1e-6;
            Parallel.For(0, height, y =>
            {
                int row = y * stride;
                for (int x = 0; x + 3 < stride; x += 4)
                {
                    int i = row + x;
                    dst[i] = ToneByte(srcPixels[i], brightness, contrast, gamma, skipGamma);
                    dst[i + 1] = ToneByte(srcPixels[i + 1], brightness, contrast, gamma, skipGamma);
                    dst[i + 2] = ToneByte(srcPixels[i + 2], brightness, contrast, gamma, skipGamma);
                    dst[i + 3] = srcPixels[i + 3];
                }
            });
            return dst;
        }

        private static byte ToneByte(byte input, double brightness, double contrast, double gamma, bool skipGamma)
        {
            double v = (input - 128.0) * contrast + 128.0 + brightness;
            v = Math.Clamp(v, 0.0, 255.0);
            if (skipGamma)
                return (byte)Math.Round(v);

            double n = v / 255.0;
            n = Math.Pow(n, 1.0 / gamma);
            return (byte)Math.Round(Math.Clamp(n * 255.0, 0.0, 255.0));
        }

        public static BitmapSource CreatePbgraBitmap(byte[] pixels, int width, int height, int stride, double dpiX, double dpiY)
        {
            var bmp = BitmapSource.Create(width, height, dpiX, dpiY, PixelFormats.Pbgra32, null, pixels, stride);
            bmp.Freeze();
            return bmp;
        }

        /// <summary>표시/저장 포맷에 맞게 변환합니다.</summary>
        public static BitmapSource ToDisplayFormat(BitmapSource pbgra32Frozen, bool wasGray8)
        {
            if (wasGray8)
            {
                var g = new FormatConvertedBitmap(pbgra32Frozen, PixelFormats.Gray8, null, 0);
                g.Freeze();
                return g;
            }
            var c = new FormatConvertedBitmap(pbgra32Frozen, PixelFormats.Bgra32, null, 0);
            c.Freeze();
            return c;
        }
    }
}
