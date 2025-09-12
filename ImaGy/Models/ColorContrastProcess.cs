using ImaGy.Wrapper;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    /// <summary>
    /// �̹����� ���� ���� �ٺ������� �����ϰų�, ��� �̹����� ������� �ϴ� ó������ ���
    /// (�׷��̽����� ��ȯ, ����ȭ, ��Ȱȭ ��)
    /// </summary>
    public class ColorContrastProcess
    {
        /// <summary>
        /// �̹����� ������� ��ȯ�ϴ� ���� ���� �޼����Դϴ�.
        /// ����ȭ, ���� ����ȭ �� ��� �̹����� �ʿ��� ó���� ����
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
        /// ������ �Ӱ谪���� ����ȭ�� �����մϴ�.
        /// �÷� �̹����� ������ ���� ������� ��ȯ�� �� ó��
        /// </summary>
        public BitmapSource ApplyBinarization(BitmapSource source, int threshold)
        {
            BitmapSource graySource = ToGrayscale(source);

            return BitmapProcessorHelper.ApplyEffect(graySource, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, threshold);
            });
        }

        /// <summary>
        /// ���� �˰����� ����� ����ȭ�� ����
        /// ����Ƽ�� �Լ��� �Ӱ谪���� -1�� �����Ͽ� ���� �˰��� ����� ����
        /// </summary>
        public BitmapSource ApplyOtsuBinarization(BitmapSource source)
        {
            BitmapSource graySource = ToGrayscale(source);

            return BitmapProcessorHelper.ApplyEffect(graySource, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyBinarization(pixelPtr, width, height, stride, -1);
            });
        }

        /// <summary>
        /// ��� �̹����� ���� ������׷� ��Ȱȭ�� ����
        /// </summary>
        public BitmapSource ApplyEqualization(BitmapSource source)
        {
            BitmapSource graySource = ToGrayscale(source);

            return BitmapProcessorHelper.ApplyEffect(graySource, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, 128);
            });
        }

        /// <summary>
        ///  �÷� �̹����� ���� ������׷� ��Ȱȭ�� �����մϴ�.
        /// BitmapProcessorHelper.ProcessBitmapSourcePixels �� �ڵ����� R, G, B ä���� �и��Ͽ� ���� ��Ȱȭ�� �����ϰ� �ٽ� ����
        /// </summary>
        public BitmapSource ApplyColorEqualization(BitmapSource source)
        {
            // ���� �̹� ��� �̹����� ������, ��� ��Ȱȭ �޼��带 ��� ȣ���մϴ�.
            if (source.Format == PixelFormats.Gray8)
            {
                return ApplyEqualization(source);
            }
            return BitmapProcessorHelper.ApplyEffect(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualization(pixelPtr, width, height, stride, 128);
            });
        }
    }
}