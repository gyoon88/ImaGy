using ImaGy.Wrapper;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Model
{
    internal class ImageProcessorSSE
    {
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
        // Edge detect process
        public BitmapSource ApplyDifferentialSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDifferentialSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplySobelSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplySobelSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyLaplacianSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyLaplacianSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Blur process
        public BitmapSource ApplyAverageBlurSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyAverageBlurSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }

        public BitmapSource ApplyGaussianBlurSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyGaussianBlurSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        // Mophorogy
        public BitmapSource ApplyDilationSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyDilationSse(pixelPtr, width, height, stride, threshold);
            }, 128);

        }

        public BitmapSource ApplyErosionSse(BitmapSource source)
        {
            return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
            {
                NativeProcessor.ApplyErosionSse(pixelPtr, width, height, stride, threshold);
            }, 128);
        }


        //// Image Matching
        //public BitmapSource ApplyNCCSse(BitmapSource source)
        //{
        //    return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
        //    {
        //        NativeProcessor.ApplyNCCSse(pixelPtr, width, height, stride, threshold);
        //    }, 128);
        //}

        //public BitmapSource ApplySADSse(BitmapSource source)
        //{
        //    return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
        //    {
        //        NativeProcessor.ApplySADSse(pixelPtr, width, height, stride, threshold);
        //    }, 128);
        //}

        //public BitmapSource ApplySSDSse(BitmapSource source)
        //{
        //    return ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride, threshold) =>
        //    {
        //        NativeProcessor.ApplySSDSse(pixelPtr, width, height, stride, threshold);
        //    }, 128);
        //}
    }
}
