using ImaGy.Wrapper;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    /// <summary>
    /// 이미지의 색상 모델을 근본적으로 변경하거나, 흑백 이미지를 대상으로 하는 처리들을 담당합니다.
    /// (그레이스케일 변환, 이진화, 평활화 등)
    /// </summary>
    public class ColorContrastProcess
    {
        /// <summary>
        /// 이미지를 흑백으로 변환하는 내부 헬퍼 메서드입니다.
        /// 이진화, 오츠 이진화 등 흑백 이미지가 필요한 처리에 재사용됩니다.
        /// </summary>
        public BitmapSource ToGrayscale(BitmapSource source)
        {
            if (source.Format == PixelFormats.Gray8)
                return source;

            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap();
            grayBitmap.BeginInit();
            grayBitmap.Source = source;
            grayBitmap.DestinationFormat = PixelFormats.Gray8;
            grayBitmap.EndInit();
            grayBitmap.Freeze();
            return grayBitmap;
        }

        /// <summary>
        /// (요청 1) 지정된 임계값으로 이진화를 적용합니다.
        /// 컬러 이미지가 들어오면 먼저 흑백으로 변환한 후 처리합니다.
        /// </summary>
        public BitmapSource ApplyBinarization(BitmapSource source, int threshold)
        {
            BitmapSource graySource = ToGrayscale(source);

            return BitmapProcessorHelper.ProcessBitmapSourcePixels(graySource, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, threshold);
            });
        }

        /// <summary>
        /// (요청 2) 오츠 알고리즘을 사용한 이진화를 적용합니다.
        /// 네이티브 함수에 임계값으로 -1을 전달하여 오츠 알고리즘 사용을 지시합니다.
        /// </summary>
        public BitmapSource ApplyOtsuBinarization(BitmapSource source)
        {
            BitmapSource graySource = ToGrayscale(source);

            return BitmapProcessorHelper.ProcessBitmapSourcePixels(graySource, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, -1);
            });
        }

        /// <summary>
        /// (요청 3 & 4) 흑백 이미지에 대한 히스토그램 평활화를 적용합니다.
        /// </summary>
        public BitmapSource ApplyEqualization(BitmapSource source)
        {
            BitmapSource graySource = ToGrayscale(source);

            return BitmapProcessorHelper.ProcessBitmapSourcePixels(graySource, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, 128);
            });
        }

        /// <summary>
        /// (요청 3) 컬러 이미지에 대한 히스토그램 평활화를 적용합니다.
        /// BitmapProcessorHelper가 자동으로 R, G, B 채널을 분리하여 각각 평활화를 적용하고 다시 병합합니다.
        /// </summary>
        public BitmapSource ApplyColorEqualization(BitmapSource source)
        {
            // 만약 이미 흑백 이미지가 들어오면, 흑백 평활화 메서드를 대신 호출합니다.
            if (source.Format == PixelFormats.Gray8)
            {
                return ApplyEqualization(source);
            }

            // 컬러 이미지의 경우, BitmapProcessorHelper가 자동으로 채널 분리/처리/병합을 수행합니다.
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, 128);
            });
        }
    }
}