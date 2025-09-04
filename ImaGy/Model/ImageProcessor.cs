using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace ImaGy.Model
{
    public class ImageProcessor
    {
        // �޸� ���·� ��ȯ�ϴ� �����Լ�
        // BitmapSource To Memory for C++ Engine 
        private BitmapSource ProcessBitmapSourcePixels(BitmapSource source, Action<IntPtr, int, int, int, byte> nativeAction, byte threshold)
        {
            // Convert to grayscale if not already, as native function expects Gray8
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap(); // ��ȯ ��ü ����
            grayBitmap.BeginInit(); // ���Ӱ� �ٲ� �� ����
            grayBitmap.Source = source; // �ʿ� �ҽ� �Ҵ�
            grayBitmap.DestinationFormat = PixelFormats.Gray8; // �ٲ� ���� ����
            grayBitmap.EndInit(); // �޸� ���ε�

            int width = grayBitmap.PixelWidth; // �����ȼ� ����
            int height = grayBitmap.PixelHeight; // �����ȼ� ����
            int stride = (width * grayBitmap.Format.BitsPerPixel + 7) / 8; // �����ȼ� * �����Ʈ / 1����Ʈ(8) �ٵ� ���ڶ�� �ȵǴϱ� �ݿø��� ���� 7�� ������.
            byte[] pixels = new byte[height * stride]; // �̹����� ����Ʈ ����
            grayBitmap.CopyPixels(pixels, stride, 0); // 

            GCHandle pinnedPixels = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject(); // AddrOfPinnedObject �� ������ �޸��� �ּҸ� ����
                nativeAction(pixelPtr, width, height, stride, threshold); // Native �Լ����� ���� 
            }
            finally
            {
                pinnedPixels.Free(); // �޸� ����
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
