using ImaGy.Wrapper;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    /// <summary>
    /// �̹����� ���� ���� �ٺ������� �����ϰų�, ��� �̹����� ������� �ϴ� ó������ ����մϴ�.
    /// (�׷��̽����� ��ȯ, ����ȭ, ��Ȱȭ ��)
    /// </summary>
    public class ColorContrastProcess
    {
        /// <summary>
        /// �̹����� ������� ��ȯ�ϴ� ���� ���� �޼����Դϴ�.
        /// ����ȭ, ���� ����ȭ �� ��� �̹����� �ʿ��� ó���� ����˴ϴ�.
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
        /// (��û 1) ������ �Ӱ谪���� ����ȭ�� �����մϴ�.
        /// �÷� �̹����� ������ ���� ������� ��ȯ�� �� ó���մϴ�.
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
        /// (��û 2) ���� �˰����� ����� ����ȭ�� �����մϴ�.
        /// ����Ƽ�� �Լ��� �Ӱ谪���� -1�� �����Ͽ� ���� �˰��� ����� �����մϴ�.
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
        /// (��û 3 & 4) ��� �̹����� ���� ������׷� ��Ȱȭ�� �����մϴ�.
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
        /// (��û 3) �÷� �̹����� ���� ������׷� ��Ȱȭ�� �����մϴ�.
        /// BitmapProcessorHelper�� �ڵ����� R, G, B ä���� �и��Ͽ� ���� ��Ȱȭ�� �����ϰ� �ٽ� �����մϴ�.
        /// </summary>
        public BitmapSource ApplyColorEqualization(BitmapSource source)
        {
            // ���� �̹� ��� �̹����� ������, ��� ��Ȱȭ �޼��带 ��� ȣ���մϴ�.
            if (source.Format == PixelFormats.Gray8)
            {
                return ApplyEqualization(source);
            }

            // �÷� �̹����� ���, BitmapProcessorHelper�� �ڵ����� ä�� �и�/ó��/������ �����մϴ�.
            return BitmapProcessorHelper.ProcessBitmapSourcePixels(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, 128);
            });
        }
    }
}