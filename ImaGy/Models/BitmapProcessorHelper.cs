using ImaGy.Wrapper;
using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;
using System.Windows;

namespace ImaGy.Models
{
    public static class BitmapProcessorHelper
    {
        /// <summary>
        /// Processes a BitmapSource by applying a native action to its pixel data.
        /// ������� ������ �ʿ� ���� �̹��� �ϳ��� ���꿡 ������ ��� ����
        /// </summary>
        public static BitmapSource ProcessBitmapSourcePixels(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            // If the image is already grayscale, process it directly.
            if (source.Format == PixelFormats.Gray8)
            {
                return ProcessGrayscaleImage(source, nativeAction);
            }
            // For all other formats (including color), process each color channel individually.
            else
            {
                return ProcessColorImage(source, nativeAction);
            }
        }

        /// <summary>
        /// Processes two images for matching. This method continues to operate on grayscale
        /// �̹��� ��Ī ���꿡�� 2���� �̹����� �޾� ����Ƽ�� ���꿡 ������ ��� ����
        /// </summary>
        public static BitmapSource ProcessTwoBitmapSourcePixels(BitmapSource source, BitmapSource template, Action<IntPtr, int, int, int, IntPtr, int, int, int, IntPtr> nativeAction)
        {
            const double maxDimension = 1500.0;
            double scale = 1.0;

            if (source.PixelWidth > maxDimension || source.PixelHeight > maxDimension)
            {
                scale = (source.PixelWidth > source.PixelHeight) ? maxDimension / source.PixelWidth : maxDimension / source.PixelHeight;
            }

            Func<BitmapSource, double, BitmapSource> scaleAndConvert = (img, s) =>
            {
                BitmapSource scaledImg = img;
                if (s < 1.0)
                {
                    scaledImg = new TransformedBitmap(img, new ScaleTransform(s, s));
                }

                FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
                grayBitmap.BeginInit();
                grayBitmap.Source = scaledImg;
                grayBitmap.DestinationFormat = PixelFormats.Gray8;
                grayBitmap.EndInit();
                return grayBitmap;
            };

            BitmapSource scaledGraySource = scaleAndConvert(source, scale);
            BitmapSource scaledGrayTemplate = scaleAndConvert(template, scale);

            int sourceWidth = scaledGraySource.PixelWidth;
            int sourceHeight = scaledGraySource.PixelHeight;
            int sourceStride = (sourceWidth * scaledGraySource.Format.BitsPerPixel + 7) / 8;
            byte[] sourcePixels = new byte[sourceHeight * sourceStride];
            scaledGraySource.CopyPixels(sourcePixels, sourceStride, 0);

            int templateWidth = scaledGrayTemplate.PixelWidth;
            int templateHeight = scaledGrayTemplate.PixelHeight;
            int templateStride = (templateWidth * scaledGrayTemplate.Format.BitsPerPixel + 7) / 8;
            byte[] templatePixels = new byte[templateHeight * templateStride];
            scaledGrayTemplate.CopyPixels(templatePixels, templateStride, 0);

            int[] coords = new int[2];
            GCHandle pinnedSourcePixels = GCHandle.Alloc(sourcePixels, GCHandleType.Pinned);
            GCHandle pinnedTemplatePixels = GCHandle.Alloc(templatePixels, GCHandleType.Pinned);
            GCHandle pinnedCoords = GCHandle.Alloc(coords, GCHandleType.Pinned);

            try
            {
                IntPtr sourcePixelPtr = pinnedSourcePixels.AddrOfPinnedObject();
                IntPtr templatePixelPtr = pinnedTemplatePixels.AddrOfPinnedObject();
                IntPtr coordPtr = pinnedCoords.AddrOfPinnedObject();

                nativeAction(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride,
                             templatePixelPtr, templateWidth, templateHeight, templateStride, coordPtr);

                int bestX_scaled = coords[0];
                int bestY_scaled = coords[1];

                int originalX = (int)(bestX_scaled / scale);
                int originalY = (int)(bestY_scaled / scale);

                return DrawBoundingBox(source, new Int32Rect(originalX, originalY, template.PixelWidth, template.PixelHeight));
            }
            finally
            {
                if (pinnedSourcePixels.IsAllocated) pinnedSourcePixels.Free();
                if (pinnedTemplatePixels.IsAllocated) pinnedTemplatePixels.Free();
                if (pinnedCoords.IsAllocated) pinnedCoords.Free();
            }
        }

        private static BitmapSource DrawBoundingBox(BitmapSource source, Int32Rect box)
        {
            var drawingSource = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

            var drawingVisual = new DrawingVisual();
            using (DrawingContext drawingContext = drawingVisual.RenderOpen())
            {
                drawingContext.DrawImage(drawingSource, new Rect(0, 0, drawingSource.PixelWidth, drawingSource.PixelHeight));
                
                Pen redPen = new Pen(Brushes.Red, 2); // Red, 2px thickness
                drawingContext.DrawRectangle(Brushes.Transparent, redPen, new Rect(box.X, box.Y, box.Width, box.Height));
            }

            var renderTarget = new RenderTargetBitmap(drawingSource.PixelWidth, drawingSource.PixelHeight, drawingSource.DpiX, drawingSource.DpiY, PixelFormats.Pbgra32);
            renderTarget.Render(drawingVisual);
            renderTarget.Freeze();

            return renderTarget;
        }


        /// <summary>
        /// ProcessBitmapSourcePicels ���� �׷��� �������� �ƴѰ�� �̹����� �޾Ƽ� �� ä�� ���� ����
        /// �� ä�� ���� C++ ���� ������ ȣ��
        /// </summary>
        private static BitmapSource ProcessColorImage(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            // 1. Standardize to Bgra32 format for consistent channel handling.
            var colorBitmap = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

            int width = colorBitmap.PixelWidth;
            int height = colorBitmap.PixelHeight;
            int colorStride = width * 4;
            byte[] allPixels = new byte[height * colorStride];
            colorBitmap.CopyPixels(allPixels, colorStride, 0);

            // 2. Split pixel data into separate channels. Alpha is preserved but not processed.
            int channelSize = width * height;
            byte[] blueChannel = new byte[channelSize];
            byte[] greenChannel = new byte[channelSize];
            byte[] redChannel = new byte[channelSize];
            byte[] alphaChannel = new byte[channelSize];

            for (int i = 0; i < allPixels.Length; i += 4)
            {
                int pixelIndex = i / 4;
                blueChannel[pixelIndex] = allPixels[i];
                greenChannel[pixelIndex] = allPixels[i + 1];
                redChannel[pixelIndex] = allPixels[i + 2];
                alphaChannel[pixelIndex] = allPixels[i + 3];
            }

            // 3. Process each color channel individually using the same native C++ function.
            ProcessSingleChannel(blueChannel, width, height, nativeAction);
            ProcessSingleChannel(greenChannel, width, height, nativeAction);
            ProcessSingleChannel(redChannel, width, height, nativeAction);

            // 4. Merge the processed channels back into a single pixel array.
            byte[] finalPixels = new byte[allPixels.Length];
            for (int i = 0; i < finalPixels.Length; i += 4)
            {
                int pixelIndex = i / 4;
                finalPixels[i] = blueChannel[pixelIndex];
                finalPixels[i + 1] = greenChannel[pixelIndex];
                finalPixels[i + 2] = redChannel[pixelIndex];
                finalPixels[i + 3] = alphaChannel[pixelIndex]; // Restore original alpha
            }

            // 5. Create a new BitmapSource from the merged, processed pixel data.
            BitmapSource result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, PixelFormats.Bgra32, null, finalPixels, colorStride);
            result.Freeze();
            return result;
        }

        /// <summary>
        /// ProcessBitmapSourcePixels ���� �׷��� �������� ��� ȣ�� ��.
        /// </summary>
        private static BitmapSource ProcessGrayscaleImage(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            int width = source.PixelWidth;
            int height = source.PixelHeight;
            int stride = (width * source.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[height * stride];
            source.CopyPixels(pixels, stride, 0);

            ProcessSingleChannel(pixels, width, height, nativeAction);

            BitmapSource result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, source.Format, null, pixels, stride);
            result.Freeze();
            return result;
        }

        /// <summary>
        ///  ProcessBitmapSourcePixels ���� �׷��� �����ϰ� �÷� �����Ϸ� ���� �� 
        ///  ProcessGreycaleImage, ProcessColorImage ���� ȣ�� �Ͽ� ���
        /// </summary>
        private static void ProcessSingleChannel(byte[] channelPixels, int width, int height, Action<IntPtr, int, int, int> nativeAction)
        {
            int stride = width; // For a single channel, stride is always equal to its width.
            GCHandle pinnedPixels = GCHandle.Alloc(channelPixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject();
                nativeAction(pixelPtr, width, height, stride);
            }
            finally
            {
                pinnedPixels.Free();
            }
        }


        /// <summary>
        ///  Kernel �� ����ϴ� �̹����� ��� �е��� ���� ���� 
        ///  �׷��� �����ϰ� �÷� �����Ϸ� ���� �� �� �÷�, �׷��� �е� �޼��带 ȣ��
        /// </summary>
        public static BitmapSource ProcessBitmapSourcePixelsWithPadding(BitmapSource source, int kernelSize, Action<IntPtr, int, int, int> nativeAction)
        {
            if (source.Format == PixelFormats.Gray8)
            {
                return ProcessGrayscaleImageWithPadding(source, kernelSize, nativeAction);
            }
            else
            {
                return ProcessColorImageWithPadding(source, kernelSize, nativeAction);
            }
        }

        /// <summary>
        ///  Kernel �� ����ϴ� �׷��� �̹����� ��� �е��� ���� ���� 
        ///  ProcessBitmapSourcePixelsWithPadding �Լ����� ȣ��
        /// </summary>
        /// 
        public static BitmapSource ProcessGrayscaleImageWithPadding(BitmapSource source, int kernelSize, Action<IntPtr, int, int, int> nativeAction)
        {
            int width = source.PixelWidth;
            int height = source.PixelHeight;
            int stride = (width * source.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[height * stride];
            source.CopyPixels(pixels, stride, 0);

            // Pad the single channel
            byte[] paddedPixels = PadChannel(pixels, width, height, kernelSize, out int paddedWidth, out int paddedHeight);
            int paddedStride = paddedWidth;

            // Process the padded channel
            ProcessSingleChannel(paddedPixels, paddedWidth, paddedHeight, nativeAction);

            // Crop the padding off the result
            byte[] resultPixels = new byte[height * stride];
            int padding = kernelSize / 2;
            for (int y = 0; y < height; y++)
            {
                int sourceIndex = (y + padding) * paddedStride + padding;
                int destIndex = y * stride;
                Buffer.BlockCopy(paddedPixels, sourceIndex, resultPixels, destIndex, width);
            }

            BitmapSource result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, source.Format, null, resultPixels, stride);
            result.Freeze();
            return result;
        }
        /// <summary>
        ///  Kernel �� ����ϴ� �÷� �̹����� ��� �е��� ���� ���� 
        ///  ProcessBitmapSourcePixelsWithPadding ���� ȣ��
        /// </summary>
        private static BitmapSource ProcessColorImageWithPadding(BitmapSource source, int kernelSize, Action<IntPtr, int, int, int> nativeAction)
        {
            var colorBitmap = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

            int width = colorBitmap.PixelWidth;
            int height = colorBitmap.PixelHeight;
            int colorStride = width * 4;
            byte[] allPixels = new byte[height * colorStride];
            colorBitmap.CopyPixels(allPixels, colorStride, 0);

            int channelSize = width * height;
            byte[] blueChannel = new byte[channelSize];
            byte[] greenChannel = new byte[channelSize];
            byte[] redChannel = new byte[channelSize];
            byte[] alphaChannel = new byte[channelSize];

            for (int i = 0; i < allPixels.Length; i += 4)
            {
                int pixelIndex = i / 4;
                blueChannel[pixelIndex] = allPixels[i];
                greenChannel[pixelIndex] = allPixels[i + 1];
                redChannel[pixelIndex] = allPixels[i + 2];
                alphaChannel[pixelIndex] = allPixels[i + 3];
            }

            // Pad and process each channel
            byte[] PadAndProcess(byte[] channel)
            {
                byte[] paddedChannel = PadChannel(channel, width, height, kernelSize, out int paddedWidth, out int paddedHeight);
                ProcessSingleChannel(paddedChannel, paddedWidth, paddedHeight, nativeAction);
                return paddedChannel;
            }

            byte[] paddedBlue = PadAndProcess(blueChannel);
            byte[] paddedGreen = PadAndProcess(greenChannel);
            byte[] paddedRed = PadAndProcess(redChannel);

            // Crop and merge
            int padding = kernelSize / 2;
            int paddedWidth = width + 2 * padding;
            byte[] finalPixels = new byte[allPixels.Length];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int paddedIndex = (y + padding) * paddedWidth + (x + padding);
                    int finalIndex = (y * width + x) * 4;

                    finalPixels[finalIndex] = paddedBlue[paddedIndex];
                    finalPixels[finalIndex + 1] = paddedGreen[paddedIndex];
                    finalPixels[finalIndex + 2] = paddedRed[paddedIndex];
                    finalPixels[finalIndex + 3] = alphaChannel[y * width + x];
                }
            }

            BitmapSource result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, PixelFormats.Bgra32, null, finalPixels, colorStride);
            result.Freeze();
            return result;
        }

        private static byte[] PadChannel(byte[] channel, int width, int height, int kernelSize, out int paddedWidth, out int paddedHeight)
        {
            int padding = kernelSize / 2;
            paddedWidth = width + 2 * padding;
            paddedHeight = height + 2 * padding;
            byte[] paddedChannel = new byte[paddedWidth * paddedHeight];

            // 1. Copy original image to the center
            for (int y = 0; y < height; y++)
            {
                int sourceIndex = y * width;
                int destIndex = (y + padding) * paddedWidth + padding;
                Buffer.BlockCopy(channel, sourceIndex, paddedChannel, destIndex, width);
            }

            // 2. Pad top and bottom edges (edge pixel replication)
            for (int x = 0; x < width; x++)
            {
                for (int p = 0; p < padding; p++)
                {
                    // Top edge
                    paddedChannel[p * paddedWidth + (x + padding)] = channel[x];
                    // Bottom edge
                    paddedChannel[(paddedHeight - 1 - p) * paddedWidth + (x + padding)] = channel[(height - 1) * width + x];
                }
            }

            // 3. Pad left and right edges (edge pixel replication)
            for (int y = 0; y < paddedHeight; y++)
            {
                for (int p = 0; p < padding; p++)
                {
                    // Left edge
                    paddedChannel[y * paddedWidth + p] = paddedChannel[y * paddedWidth + padding];
                    // Right edge
                    paddedChannel[y * paddedWidth + (paddedWidth - 1 - p)] = paddedChannel[y * paddedWidth + (paddedWidth - 1 - padding)];
                }
            }

            return paddedChannel;
        }
    }
    //public static BitmapSource ProcessTwoBitmapSourcePixels(BitmapSource source, BitmapSource template, Action<IntPtr, int, int, int, IntPtr, int, int, int> nativeAction)
    //{
    //    // ���� �̹����� �׷��̽����Ϸ� ��ȯ�� .
    //    FormatConvertedBitmap graySourceBitmap = new FormatConvertedBitmap();
    //    graySourceBitmap.BeginInit();
    //    graySourceBitmap.Source = source;
    //    graySourceBitmap.DestinationFormat = PixelFormats.Gray8;
    //    graySourceBitmap.EndInit();

    //    int sourceWidth = graySourceBitmap.PixelWidth;
    //    int sourceHeight = graySourceBitmap.PixelHeight;
    //    int sourceStride = (sourceWidth * graySourceBitmap.Format.BitsPerPixel + 7) / 8;
    //    byte[] sourcePixels = new byte[sourceHeight * sourceStride];
    //    graySourceBitmap.CopyPixels(sourcePixels, sourceStride, 0);

    //    // ���ø� �̹����� �׷��� �����Ϸ� ��ȯ
    //    FormatConvertedBitmap grayTemplateBitmap = new FormatConvertedBitmap();
    //    grayTemplateBitmap.BeginInit();
    //    grayTemplateBitmap.Source = template;
    //    grayTemplateBitmap.DestinationFormat = PixelFormats.Gray8;
    //    grayTemplateBitmap.EndInit();

    //    int templateWidth = grayTemplateBitmap.PixelWidth;
    //    int templateHeight = grayTemplateBitmap.PixelHeight;
    //    int templateStride = (templateWidth * grayTemplateBitmap.Format.BitsPerPixel + 7) / 8;
    //    byte[] templatePixels = new byte[templateHeight * templateStride];
    //    grayTemplateBitmap.CopyPixels(templatePixels, templateStride, 0);

    //    // ���� �̹����� ���ø� �̹����� �޸� �ּҸ� ����
    //    GCHandle pinnedSourcePixels = GCHandle.Alloc(sourcePixels, GCHandleType.Pinned);
    //    GCHandle pinnedTemplatePixels = GCHandle.Alloc(templatePixels, GCHandleType.Pinned);

    //    // �޸� ������ Native Action C++ �� ������ �ѱ�
    //    try
    //    {
    //        IntPtr sourcePixelPtr = pinnedSourcePixels.AddrOfPinnedObject();
    //        IntPtr templatePixelPtr = pinnedTemplatePixels.AddrOfPinnedObject();

    //        nativeAction(sourcePixelPtr, sourceWidth, sourceHeight, sourceStride,
    //                     templatePixelPtr, templateWidth, templateHeight, templateStride);

    //        BitmapSource resultBitmap = BitmapSource.Create(
    //            sourceWidth,
    //            sourceHeight,
    //            source.DpiX,
    //            source.DpiY,
    //            PixelFormats.Gray8,
    //            null,
    //            sourcePixels,
    //            sourceStride);

    //        resultBitmap.Freeze();
    //        return resultBitmap;
    //    }
    //    finally
    //    {
    //        if (pinnedSourcePixels.IsAllocated) pinnedSourcePixels.Free();
    //        if (pinnedTemplatePixels.IsAllocated) pinnedTemplatePixels.Free();
    //    }
    //}


}
