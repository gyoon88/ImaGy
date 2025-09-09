using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Collections.Generic; // Dictionary를 위해 추가
using ImaGy.Wrapper;

namespace ImaGy.Models
{
    public class ServeHistogram
    {
        public static int[] CalculateGrayscaleHistogram(BitmapSource source)
        {
            if (source == null) return new int[256];

            BitmapSource bitmapToUse;

            // 1. 흑백이면 그대로, 아니면 Gray8로 변환
            if (source.Format == PixelFormats.Gray8)
            {
                bitmapToUse = source;
            }
            else
            {
                FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
                grayBitmap.BeginInit();
                grayBitmap.Source = source;
                grayBitmap.DestinationFormat = PixelFormats.Gray8;
                grayBitmap.EndInit();
                bitmapToUse = grayBitmap;
            }

            int width = bitmapToUse.PixelWidth;
            int height = bitmapToUse.PixelHeight;
            int stride = width;  // Gray8 = 1byte per pixel
            byte[] pixels = new byte[height * stride];
            bitmapToUse.CopyPixels(pixels, stride, 0);

            int[] histogram = new int[256];

            // 2. C++ 네이티브 함수 호출
            unsafe
            {
                fixed (byte* pPixels = pixels)
                fixed (int* pHist = histogram)
                {
                    // The GCHandle is no longer needed.
                    // Pass the pointers directly to the native function.
                    IntPtr ptr = (IntPtr)pPixels;
                    NativeProcessor.ApplyHistogram(ptr, width, height, stride, pHist);
                }
            }
            return histogram;
        }

        /// <summary>
        /// 컬러 이미지의 R, G, B, A 각 채널에 대한 히스토그램을 계산
        /// </summary>
        /// <returns>"R", "G", "B", "A"를 키로 갖는 히스토그램 Dictionary</returns>
        public static Dictionary<string, int[]> CalculateColorHistograms(BitmapSource source)
        {
            if (source == null) return new Dictionary<string, int[]>();
            FormatConvertedBitmap bitmapToUse = new FormatConvertedBitmap();
            bitmapToUse.BeginInit();
            bitmapToUse.Source = source;
            bitmapToUse.DestinationFormat = PixelFormats.Bgra32;
            bitmapToUse.EndInit();

            int width = bitmapToUse.PixelWidth;
            int height = bitmapToUse.PixelHeight;

            // Bgra32는 픽셀당 4바이트(32비트)를 사용
            int stride = width * 4;
            byte[] allPixels = new byte[height * stride];
            bitmapToUse.CopyPixels(allPixels, stride, 0);

            // 각 채널별로 픽셀 데이터를 분리할 배열을 준비
            byte[] blueChannel = new byte[width * height];
            byte[] greenChannel = new byte[width * height];
            byte[] redChannel = new byte[width * height];
            byte[] alphaChannel = new byte[width * height];

            for (int i = 0; i < allPixels.Length; i += 4)
            {
                int pixelIndex = i / 4;
                blueChannel[pixelIndex] = allPixels[i];     
                greenChannel[pixelIndex] = allPixels[i + 1]; 
                redChannel[pixelIndex] = allPixels[i + 2];   
                alphaChannel[pixelIndex] = allPixels[i + 3]; // Alpha
            }

            var histograms = new Dictionary<string, int[]>();

            // 각 채널에 대해 기존 C++ 함수를 호출하여 히스토그램을 계산
            histograms["B"] = CalculateHistogramForChannel(blueChannel, width, height);
            histograms["G"] = CalculateHistogramForChannel(greenChannel, width, height);
            histograms["R"] = CalculateHistogramForChannel(redChannel, width, height);
            histograms["A"] = CalculateHistogramForChannel(alphaChannel, width, height);

            return histograms;
        }

        // 네이티브 함수 호출 로직을 별도 메서드로 분리하여 코드 중복을 줄입니다.
        private static int[] CalculateHistogramForChannel(byte[] pixels, int width, int height)
        {
            int[] histogram = new int[256];
            // 단일 채널이므로 stride는 이미지의 width와 같습니다.
            int stride = width;

            unsafe
            {
                fixed (byte* pPixels = pixels)
                fixed (int* pHist = histogram)
                {
                    IntPtr ptr = (IntPtr)pPixels;
                    NativeProcessor.ApplyHistogram(ptr, width, height, stride, pHist);
                }
            }
            return histogram;
        }
    }
}