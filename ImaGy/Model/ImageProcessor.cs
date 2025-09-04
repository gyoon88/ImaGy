using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace ImaGy.Model
{
    public class ImageProcessor
    {
        // 메모리 형태로 변환하는 고차함수
        // BitmapSource To Memory for C++ Engine 
        private BitmapSource ProcessBitmapSourcePixels(BitmapSource source, Action<IntPtr, int, int, int, byte> nativeAction, byte threshold)
        {
            // Convert to grayscale if not already, as native function expects Gray8
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap(); // 변환 객체 생성
            grayBitmap.BeginInit(); // 새롭게 바꿀 맵 생성
            grayBitmap.Source = source; // 맵에 소스 할당
            grayBitmap.DestinationFormat = PixelFormats.Gray8; // 바꿀 포멧 전달
            grayBitmap.EndInit(); // 메모리 업로드

            int width = grayBitmap.PixelWidth; // 가로픽셀 개수
            int height = grayBitmap.PixelHeight; // 세로픽셀 개수
            int stride = (width * grayBitmap.Format.BitsPerPixel + 7) / 8; // 가로픽셀 * 색상비트 / 1바이트(8) 근데 모자라면 안되니까 반올림을 위해 7을 더해줌.
            byte[] pixels = new byte[height * stride]; // 이미지의 바이트 정보
            grayBitmap.CopyPixels(pixels, stride, 0); // 

            GCHandle pinnedPixels = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject(); // AddrOfPinnedObject 로 고정된 메모리의 주소를 얻음
                nativeAction(pixelPtr, width, height, stride, threshold); // Native 함수에게 전달 
            }
            finally
            {
                pinnedPixels.Free(); // 메모리 해제
            }

            // Create a new BitmapSource from the modified pixel data
            BitmapSource result = BitmapSource.Create(
                width,
                height,
                grayBitmap.DpiX,
                grayBitmap.DpiY,
                grayBitmap.Format,
                null, // Color palette
                pixels,  // Use the modified managed array
                stride);

            result.Freeze(); // Recommended for performance and thread safety
            return result;
        }
        
        
        // Colour | contrast
        public BitmapSource ApplyBinarization(BitmapSource source, byte threshold)
        {
            // Convert to grayscale
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
            grayBitmap.BeginInit();
            grayBitmap.Source = source;
            grayBitmap.DestinationFormat = PixelFormats.Gray8;
            grayBitmap.EndInit();

            // Get pixel data
            int stride = (grayBitmap.PixelWidth * grayBitmap.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[grayBitmap.PixelHeight * stride];
            grayBitmap.CopyPixels(pixels, stride, 0);

            // Apply threshold
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = pixels[i] < threshold ? (byte)0 : (byte)255;
            }

            // Create new bitmap
            BitmapSource result = BitmapSource.Create(
                grayBitmap.PixelWidth,
                grayBitmap.PixelHeight,
                grayBitmap.DpiX,
                grayBitmap.DpiY,
                grayBitmap.Format,
                null,
                pixels,
                stride);

            result.Freeze(); // Improve performance
            return result;
        }

        public BitmapSource ApplyEqualization(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        // Edge detect process
        public BitmapSource ApplyDifferential(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDifferential(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplySobel(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplySobel(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyLaplacian(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyLaplacian(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Blur process
        public BitmapSource ApplyAverageBlur(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyAverageBlur(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyGaussianBlur(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyGaussianBlur(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Mophorogy
        public BitmapSource ApplyDilation(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDilation(pixelPtr, width, height, stride, threshold);
            }, 128);

        }

        public BitmapSource ApplyErosion(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyErosion(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Image Matching
        public BitmapSource ApplyNCC(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyNCC(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplySAD(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplySAD(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplySSD(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplySSD(pixelPtr, width, height, stride, threshold);
            }, 128);
        }
    }

}
