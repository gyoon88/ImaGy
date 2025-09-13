using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public static class BitmapProcessorHelper
    {
        // =================================================================
        // --- 범용 이미지 처리 헬퍼 ---
        // =================================================================

        /// <summary>
        /// [일반 처리용] 패딩 없이 네이티브 액션을 적용
        /// </summary>
        public static BitmapSource ApplyEffect(BitmapSource source, Action<IntPtr, int, int, int> nativeAction)
        {
            if (source.Format != PixelFormats.Gray8 && source.Format != PixelFormats.Bgra32)
                throw new ArgumentException("입력 이미지는 Gray8 또는 Bgra32 포맷이어야 합니다.", nameof(source));

            int width = source.PixelWidth;
            int height = source.PixelHeight;
            int stride = (width * source.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[height * stride];
            source.CopyPixels(pixels, stride, 0);

            GCHandle pinnedPixels = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject();
                nativeAction(pixelPtr, width, height, stride);
            }
            finally
            {
                pinnedPixels.Free();
            }

            var result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, source.Format, null, pixels, stride);
            result.Freeze();
            return result;
        }

        /// <summary>
        /// [커널 처리용] 패딩/크롭을 포함하여 네이티브 액션을 적용합니다.
        /// </summary>
        public static BitmapSource ApplyKernelEffect(BitmapSource source, int kernelSize, Action<IntPtr, int, int, int> nativeAction)
        {
            if (source.Format != PixelFormats.Gray8 && source.Format != PixelFormats.Bgra32)
                throw new ArgumentException("입력 이미지는 Gray8 또는 Bgra32 포맷이어야 합니다.", nameof(source));

            int width = source.PixelWidth;
            int height = source.PixelHeight;
            int pixelSize = source.Format.BitsPerPixel / 8;
            int stride = width * pixelSize;
            byte[] originalPixels = new byte[height * stride];
            source.CopyPixels(originalPixels, stride, 0);

            byte[] paddedPixels = PadBuffer(originalPixels, width, height, kernelSize, pixelSize, out int paddedWidth, out int paddedHeight);
            int paddedStride = paddedWidth * pixelSize;

            GCHandle pinnedPixels = GCHandle.Alloc(paddedPixels, GCHandleType.Pinned);
            try
            {
                IntPtr pixelPtr = pinnedPixels.AddrOfPinnedObject();
                nativeAction(pixelPtr, paddedWidth, paddedHeight, paddedStride);
            }
            finally
            {
                pinnedPixels.Free();
            }

            byte[] resultPixels = CropBuffer(paddedPixels, width, height, kernelSize, pixelSize);

            var result = BitmapSource.Create(width, height, source.DpiX, source.DpiY, source.Format, null, resultPixels, stride);
            result.Freeze();
            return result;
        }

        // =================================================================
        // --- 패딩/크롭 전용 헬퍼 ---
        // =================================================================

        private static byte[] PadBuffer(byte[] originalPixels, int width, int height, int kernelSize, int pixelSize, out int paddedWidth, out int paddedHeight)
        {
            int padding = kernelSize / 2;
            paddedWidth = width + 2 * padding;
            paddedHeight = height + 2 * padding;
            byte[] paddedPixels = new byte[paddedWidth * paddedHeight * pixelSize];
            int stride = width * pixelSize;
            int paddedStride = paddedWidth * pixelSize;

            for (int y = 0; y < height; y++)
            {
                Buffer.BlockCopy(originalPixels, y * stride, paddedPixels, (y + padding) * paddedStride + padding * pixelSize, stride);
            }
            for (int p = 0; p < padding; p++)
            {
                Buffer.BlockCopy(paddedPixels, (padding) * paddedStride, paddedPixels, p * paddedStride, paddedStride);
                Buffer.BlockCopy(paddedPixels, (paddedHeight - padding - 1) * paddedStride, paddedPixels, (paddedHeight - p - 1) * paddedStride, paddedStride);
            }
            for (int y = 0; y < paddedHeight; y++)
            {
                for (int p = 0; p < padding; p++)
                {
                    Buffer.BlockCopy(paddedPixels, y * paddedStride + padding * pixelSize, paddedPixels, y * paddedStride + p * pixelSize, pixelSize);
                    Buffer.BlockCopy(paddedPixels, y * paddedStride + (paddedWidth - padding - 1) * pixelSize, paddedPixels, y * paddedStride + (paddedWidth - p - 1) * pixelSize, pixelSize);
                }
            }
            return paddedPixels;
        }

        private static byte[] CropBuffer(byte[] paddedPixels, int originalWidth, int originalHeight, int kernelSize, int pixelSize)
        {
            int padding = kernelSize / 2;
            int originalStride = originalWidth * pixelSize;
            int paddedWidth = originalWidth + 2 * padding;
            int paddedStride = paddedWidth * pixelSize;
            byte[] croppedPixels = new byte[originalWidth * originalHeight * pixelSize];

            for (int y = 0; y < originalHeight; y++)
            {
                int sourceIndex = (y + padding) * paddedStride + padding * pixelSize;
                int destIndex = y * originalStride;
                Buffer.BlockCopy(paddedPixels, sourceIndex, croppedPixels, destIndex, originalStride);
            }
            return croppedPixels;
        }

        // =================================================================
        // --- 템플릿 매칭 전용 헬퍼 ---
        // =================================================================

        public static BitmapSource ProcessTwoBitmapSourcePixels(BitmapSource source, BitmapSource template, Action<IntPtr, int, int, int, IntPtr, int, int, int, IntPtr> nativeAction)
        {
            const double Max_Dimension = 1200.0;
            double scale = 1.0;

            if (source.PixelWidth > Max_Dimension || source.PixelHeight > Max_Dimension)
            {
                scale = (source.PixelWidth > source.PixelHeight) ? Max_Dimension / source.PixelWidth : Max_Dimension / source.PixelHeight;
            }

            Func<BitmapSource, double, BitmapSource> scaleAndConvert = (img, s) =>
            {
                BitmapSource scaledImg = (s < 1.0) ? new TransformedBitmap(img, new ScaleTransform(s, s)) : img;
                if (scaledImg.Format == PixelFormats.Gray8) return scaledImg;
                return new FormatConvertedBitmap(scaledImg, PixelFormats.Gray8, null, 0);
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
                Pen redPen = new Pen(Brushes.Red, 2);
                // 50% 투명도를 가진 빨간색 브러시 (A:128, R:255, G:0, B:0)
                Brush semiTransparentRedBrush = new SolidColorBrush(Color.FromArgb(64, 255, 0, 0));


                // 3. 사각형을 그립니다. (내부를 redBrush로 채우고, 테두리는 redPen으로 그립니다)
                drawingContext.DrawRectangle(semiTransparentRedBrush, redPen, new Rect(box.X, box.Y, box.Width, box.Height));
            }
            var renderTarget = new RenderTargetBitmap(drawingSource.PixelWidth, drawingSource.PixelHeight, drawingSource.DpiX, drawingSource.DpiY, PixelFormats.Pbgra32);
            renderTarget.Render(drawingVisual);
            renderTarget.Freeze();
            return renderTarget;
        }
    }
}