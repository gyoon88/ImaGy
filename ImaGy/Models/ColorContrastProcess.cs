using ImaGy.Wrapper;
using OpenCvSharp;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    /// <summary>
    /// 이미지의 색상 모델을 근본적으로 변경하거나, 흑백 이미지를 대상으로 하는 처리들을 담당
    /// (그레이스케일 변환, 이진화, 평활화 등)
    /// </summary>
    public class ColorContrastProcess
    {
        /// <summary>
        /// 이미지를 흑백으로 변환하는 내부 헬퍼 메서드
        /// 이진화, 오츠 이진화 등 흑백 이미지가 필요한 처리에 재사용
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
        /// 지정된 임계값으로 이진화를 적용
        /// 컬러 이미지가 들어오면 먼저 흑백으로 변환한 후 처리
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
        ///// GMM(Gaussian Mixture Model) 클러스터링을 순수 C# 코드로 이미지에 적용합니다.
        ///// 색상(BGR)과 공간(x, y) 정보를 함께 사용하여 영역을 분할합니다.
        ///// </summary>
        ///// <param name="source">처리할 원본 BitmapSource.</param>
        ///// <param name="numClusters">분할할 클러스터의 개수 (k).</param>
        ///// <param name="spatialWeight">공간 정보의 가중치 (0.0 ~ 1.0).</param>
        ///// <returns>GMM 분할 결과가 적용된 새로운 BitmapSource.</returns>
        //public BitmapSource ApplyGmmSegmentation(BitmapSource source, int numClusters, float spatialWeight)
        //{
        //    // GMM은 컬러 이미지를 대상으로 하므로 흑백 이미지는 처리하지 않음.
        //    if (source.Format == PixelFormats.Gray8)
        //    {
        //        return source;
        //    }

        //    // OpenCvSharp의 Mat 객체들은 C++ 메모리를 사용하므로, using 또는 Dispose()로 관리해야 합니다.
        //    Mat sourceMat = null;
        //    Mat samples = null;
        //    Mat labels = null;
        //    Mat segmentedLabels = null;
        //    Mat normalizedLabels = null;
        //    Mat coloredResult = null;

        //    try
        //    {
        //        // 1. 입력 BitmapSource를 OpenCvSharp의 Mat 객체로 변환
        //        sourceMat = BitmapSourceConverter.ToMat(source);
        //        // GMM이 BGR을 기준으로 하므로, BGRA 포맷인 경우 BGR로 변환
        //        if (sourceMat.Channels() == 4)
        //        {
        //            Cv2.CvtColor(sourceMat, sourceMat, ColorConversionCodes.BGRA2BGR);
        //        }

        //        int height = sourceMat.Rows;
        //        int width = sourceMat.Cols;
        //        int numPixels = height * width;

        //        // 2. 특징 벡터 생성 (B, G, R, x*w, y*w)
        //        samples = new Mat(numPixels, 5, MatType.CV_32FC1);
        //        for (int y = 0; y < height; y++)
        //        {
        //            for (int x = 0; x < width; x++)
        //            {
        //                Vec3b pixel = sourceMat.At<Vec3b>(y, x);
        //                int pixelIndex = y * width + x;

        //                // samples Mat에 값 설정
        //                samples.Set(pixelIndex, 0, (float)pixel.Item0); // B
        //                samples.Set(pixelIndex, 1, (float)pixel.Item1); // G
        //                samples.Set(pixelIndex, 2, (float)pixel.Item2); // R
        //                samples.Set(pixelIndex, 3, (float)x * spatialWeight);
        //                samples.Set(pixelIndex, 4, (float)y * spatialWeight);
        //            }
        //        }

        //        // 3. GMM 모델 생성 및 학습
        //        using (var gmm = EM.Create())
        //        {
        //            gmm.ClustersNumber = numClusters;
        //            gmm.TermCriteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.Count, 100, 0.1);

        //            labels = new Mat();
        //            gmm.TrainEM(samples, null, labels, null);
        //        }

        //        // 4. 결과를 시각화 가능한 이미지로 변환
        //        segmentedLabels = labels.Reshape(1, height);
        //        normalizedLabels = new Mat();
        //        segmentedLabels.ConvertTo(normalizedLabels, MatType.CV_8UC1, 255.0 / (numClusters - 1));

        //        coloredResult = new Mat();
        //        Cv2.ApplyColorMap(normalizedLabels, coloredResult, ColormapTypes.Jet);

        //        // 5. 최종 결과 Mat을 BitmapSource로 변환하여 반환
        //        return BitmapSourceConverter.ToBitmapSource(coloredResult);
        //    }
        //    finally
        //    {
        //        // C# 가비지 컬렉터가 관리하지 않는 C++ 메모리를 명시적으로 해제합니다.
        //        sourceMat?.Dispose();
        //        samples?.Dispose();
        //        labels?.Dispose();
        //        segmentedLabels?.Dispose();
        //        normalizedLabels?.Dispose();
        //        coloredResult?.Dispose();
        //    }
        //}
    


        /// <summary>
        /// 오츠 알고리즘을 사용한 이진화를 적용
        /// 네이티브 함수에 임계값으로 -1을 전달하여 오츠 알고리즘 사용을 지시
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
        /// 흑백 이미지에 대한 히스토그램 평활화를 적용
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
        ///  컬러 이미지에 대한 히스토그램 평활화를 적용
        /// BitmapProcessorHelper.ProcessBitmapSourcePixels 가 자동으로 R, G, B 채널을 분리하여 각각 평활화를 적용하고 다시 병합
        /// </summary>
        public BitmapSource ApplyColorEqualization(BitmapSource source)
        {
            // 흑백 이미지 입력 시 흑백 평활화로 예외처리
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