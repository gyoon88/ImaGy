using ImaGy.Wrapper;
using System.Windows;
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
        /// �̹����� ������� ��ȯ�ϴ� ���� ���� �޼���
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
        /// ������ �Ӱ谪���� ����ȭ�� ����
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
        public BitmapSource ApplyKMeansClustering(BitmapSource source, int k, int iteration, bool location, bool isColor )
        {
            if (!isColor)
            {
                return source;
            }
            return BitmapProcessorHelper.ApplyEffect(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyKMeansClustering(pixelPtr, width, height, stride, k, iteration, location);
            });
        }



        ///// <summary>
        ///// GMM(Gaussian Mixture Model) Ŭ�����͸��� ���� C# �ڵ�� �̹����� �����մϴ�.
        ///// ����(BGR)�� ����(x, y) ������ �Բ� ����Ͽ� ������ �����մϴ�.
        ///// </summary>
        ///// <param name="source">ó���� ���� BitmapSource.</param>
        ///// <param name="numClusters">������ Ŭ�������� ���� (k).</param>
        ///// <param name="spatialWeight">���� ������ ����ġ (0.0 ~ 1.0).</param>
        ///// <returns>GMM ���� ����� ����� ���ο� BitmapSource.</returns>
        //public BitmapSource ApplyGmmSegmentation(BitmapSource source, int numClusters, float spatialWeight)
        //{
        //    // GMM�� �÷� �̹����� ������� �ϹǷ� ��� �̹����� ó������ ����.
        //    if (source.Format == PixelFormats.Gray8)
        //    {
        //        return source;
        //    }

        //    // OpenCvSharp�� Mat ��ü���� C++ �޸𸮸� ����ϹǷ�, using �Ǵ� Dispose()�� �����ؾ� �մϴ�.
        //    Mat sourceMat = null;
        //    Mat samples = null;
        //    Mat labels = null;
        //    Mat segmentedLabels = null;
        //    Mat normalizedLabels = null;
        //    Mat coloredResult = null;

        //    try
        //    {
        //        // 1. �Է� BitmapSource�� OpenCvSharp�� Mat ��ü�� ��ȯ
        //        sourceMat = BitmapSourceConverter.ToMat(source);
        //        // GMM�� BGR�� �������� �ϹǷ�, BGRA ������ ��� BGR�� ��ȯ
        //        if (sourceMat.Channels() == 4)
        //        {
        //            Cv2.CvtColor(sourceMat, sourceMat, ColorConversionCodes.BGRA2BGR);
        //        }

        //        int height = sourceMat.Rows;
        //        int width = sourceMat.Cols;
        //        int numPixels = height * width;

        //        // 2. Ư¡ ���� ���� (B, G, R, x*w, y*w)
        //        samples = new Mat(numPixels, 5, MatType.CV_32FC1);
        //        for (int y = 0; y < height; y++)
        //        {
        //            for (int x = 0; x < width; x++)
        //            {
        //                Vec3b pixel = sourceMat.At<Vec3b>(y, x);
        //                int pixelIndex = y * width + x;

        //                // samples Mat�� �� ����
        //                samples.Set(pixelIndex, 0, (float)pixel.Item0); // B
        //                samples.Set(pixelIndex, 1, (float)pixel.Item1); // G
        //                samples.Set(pixelIndex, 2, (float)pixel.Item2); // R
        //                samples.Set(pixelIndex, 3, (float)x * spatialWeight);
        //                samples.Set(pixelIndex, 4, (float)y * spatialWeight);
        //            }
        //        }

        //        // 3. GMM �� ���� �� �н�
        //        using (var gmm = EM.Create())
        //        {
        //            gmm.ClustersNumber = numClusters;
        //            gmm.TermCriteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.Count, 100, 0.1);

        //            labels = new Mat();
        //            gmm.TrainEM(samples, null, labels, null);
        //        }

        //        // 4. ����� �ð�ȭ ������ �̹����� ��ȯ
        //        segmentedLabels = labels.Reshape(1, height);
        //        normalizedLabels = new Mat();
        //        segmentedLabels.ConvertTo(normalizedLabels, MatType.CV_8UC1, 255.0 / (numClusters - 1));

        //        coloredResult = new Mat();
        //        Cv2.ApplyColorMap(normalizedLabels, coloredResult, ColormapTypes.Jet);

        //        // 5. ���� ��� Mat�� BitmapSource�� ��ȯ�Ͽ� ��ȯ
        //        return BitmapSourceConverter.ToBitmapSource(coloredResult);
        //    }
        //    finally
        //    {
        //        // C# ������ �÷��Ͱ� �������� �ʴ� C++ �޸𸮸� ���������� �����մϴ�.
        //        sourceMat?.Dispose();
        //        samples?.Dispose();
        //        labels?.Dispose();
        //        segmentedLabels?.Dispose();
        //        normalizedLabels?.Dispose();
        //        coloredResult?.Dispose();
        //    }
        //}
    


        /// <summary>
        /// ���� �˰������� ����� ����ȭ�� ����
        /// ����Ƽ�� �Լ��� �Ӱ谪���� -1�� �����Ͽ� ���� �˰����� ����� ����
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
        ///  �÷� �̹����� ���� ������׷� ��Ȱȭ�� ����
        /// BitmapProcessorHelper.ProcessBitmapSourcePixels �� �ڵ����� R, G, B ä���� �и��Ͽ� ���� ��Ȱȭ�� �����ϰ� �ٽ� ����
        /// </summary>
        public BitmapSource ApplyColorEqualization(BitmapSource source)
        {
            // ��� �̹��� �Է� �� ��� ��Ȱȭ�� ����ó��
            if (source.Format == PixelFormats.Gray8)
            {
                return ApplyEqualization(source);
            }
            return BitmapProcessorHelper.ApplyEffect(source, (pixelPtr, width, height, stride) =>
            {
                NativeProcessor.ApplyEqualizationColor(pixelPtr, width, height, stride, 128);
            });
        }
    }
}