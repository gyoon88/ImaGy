using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Collections.Generic;
using System.ComponentModel;
using ImaGy.Models;

namespace ImaGy.ViewModels
{
    public class HistogramViewModel : BaseViewModel
    {
        private MainViewModel mainViewModel;

        #region Public Properties for View Binding

        // 히스토그램 데이터 속성
        public int[]? R_HistogramData { get; private set; }
        public int[]? G_HistogramData { get; private set; }
        public int[]? B_HistogramData { get; private set; }
        public int[]? GrayscaleHistogramData { get; private set; }

        private int maxHistogramValue;
        public int MaxHistogramValue
        {
            get => maxHistogramValue;
            private set => SetProperty(ref maxHistogramValue, value);
        }

        private bool isColorImage;
        public bool IsColorImage
        {
            get => isColorImage;
            private set => SetProperty(ref isColorImage, value);
        }

        #endregion

        #region Public Properties for Statistics
        // 요약 통계량
        private double? mean;
        private double? std;
        private int? median;
        private int? mode;
        private int? max;
        private int? min;
        private int? range;

        public double? Mean { get => mean; private set => SetProperty(ref mean, value); }
        public double? Std { get => std; private set => SetProperty(ref std, value); }
        public int? Median { get => median; private set => SetProperty(ref median, value); }
        public int? Mode { get => mode; private set => SetProperty(ref mode, value); }
        public int? Max { get => max; private set => SetProperty(ref max, value); }
        public int? Min { get => min; private set => SetProperty(ref min, value); }
        public int? Range { get => range; private set => SetProperty(ref range, value); }
        #endregion

        public HistogramViewModel(MainViewModel mainViewModel)
        {
            this.mainViewModel = mainViewModel;
            this.mainViewModel.PropertyChanged += MainViewModel_PropertyChanged;
            UpdateAllHistograms();
        }

        private void MainViewModel_PropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.AfterImage) || e.PropertyName == nameof(MainViewModel.BeforeImage))
            {
                UpdateAllHistograms();
            }
        }

        private void UpdateAllHistograms()
        {
            if (mainViewModel == null) return;

            BitmapSource? imageSource = mainViewModel.AfterImage ?? mainViewModel.BeforeImage;
            if (imageSource != null)
            {
                if (imageSource.Format == PixelFormats.Gray8)
                {
                    IsColorImage = false;
                    GrayscaleHistogramData = ServeHistogram.CalculateGrayscaleHistogram(imageSource);
                    R_HistogramData = G_HistogramData = B_HistogramData = null;
                    MaxHistogramValue = GrayscaleHistogramData.Any() ? GrayscaleHistogramData.Max() : 0;

                    // 흑백 이미지 히스토그램으로 통계 계산
                    CalculateAndSetStatistics(GrayscaleHistogramData);
                }
                else
                {
                    IsColorImage = true;
                    var colorHistograms = ServeHistogram.CalculateColorHistograms(imageSource);

                    colorHistograms.TryGetValue("R", out var rData);
                    colorHistograms.TryGetValue("G", out var gData);
                    colorHistograms.TryGetValue("B", out var bData);

                    R_HistogramData = rData;
                    G_HistogramData = gData;
                    B_HistogramData = bData;
                    GrayscaleHistogramData = null;

                    int maxR = R_HistogramData?.Max() ?? 0;
                    int maxG = G_HistogramData?.Max() ?? 0;
                    int maxB = B_HistogramData?.Max() ?? 0;
                    MaxHistogramValue = Math.Max(maxR, Math.Max(maxG, maxB));

                    // 컬러 이미지의 경우, 그레이스케일로 변환하여 통계 계산
                    var statsHistogram = ServeHistogram.CalculateGrayscaleHistogram(imageSource);
                    CalculateAndSetStatistics(statsHistogram);
                }
            }
            else
            {
                // 이미지가 없을 경우 모든 데이터 초기화
                IsColorImage = false;
                GrayscaleHistogramData = R_HistogramData = G_HistogramData = B_HistogramData = null;
                MaxHistogramValue = 0;
                ClearStatistics();
            }

            // View가 데이터를 다시 그리도록 모든 관련 속성의 변경을 알림.
            OnPropertyChanged(nameof(IsColorImage));
            OnPropertyChanged(nameof(R_HistogramData));
            OnPropertyChanged(nameof(G_HistogramData));
            OnPropertyChanged(nameof(B_HistogramData));
            OnPropertyChanged(nameof(GrayscaleHistogramData));
        }

        /// <summary>
        /// 모든 통계량 속성을 null로 초기화합니다.
        /// </summary>
        private void ClearStatistics()
        {
            Mean = null;
            Std = null;
            Median = null;
            Mode = null;
            Max = null;
            Min = null;
            Range = null;
        }

        /// <summary>
        /// 주어진 히스토그램 데이터를 기반으로 요약 통계량을 계산하고 속성을 설정합니다.
        /// </summary>
        /// <param name="histogram">256개의 bin을 가진 히스토그램 배열</param>
        private void CalculateAndSetStatistics(int[]? histogram)
        {
            if (histogram == null || histogram.Length != 256 || histogram.Sum() == 0)
            {
                ClearStatistics();
                return;
            }

            long totalPixels = 0;
            long sumOfIntensities = 0;
            for (int i = 0; i < 256; i++)
            {
                totalPixels += histogram[i];
                sumOfIntensities += (long)i * histogram[i];
            }

            if (totalPixels == 0)
            {
                ClearStatistics();
                return;
            }

            // Mean (평균)
            double meanValue = (double)sumOfIntensities / totalPixels;
            Mean = meanValue;

            // Standard Deviation (표준편차)
            double sumOfSquaredDifferences = 0;
            for (int i = 0; i < 256; i++)
            {
                sumOfSquaredDifferences += Math.Pow(i - meanValue, 2) * histogram[i];
            }
            Std = Math.Sqrt(sumOfSquaredDifferences / totalPixels);

            // Median (중앙값)
            long cumulativeFrequency = 0;
            long medianThreshold = totalPixels / 2;
            int medianValue = 0;
            for (int i = 0; i < 256; i++)
            {
                cumulativeFrequency += histogram[i];
                if (cumulativeFrequency >= medianThreshold)
                {
                    medianValue = i;
                    break;
                }
            }
            Median = medianValue;

            // Mode (최빈값)
            int maxFrequency = 0;
            int modeValue = 0;
            for (int i = 0; i < 256; i++)
            {
                if (histogram[i] > maxFrequency)
                {
                    maxFrequency = histogram[i];
                    modeValue = i;
                }
            }
            Mode = modeValue;

            // Min & Max Intensity (최소/최대 밝기)
            int minValue = -1;
            int maxValue = -1;
            for (int i = 0; i < 256; i++)
            {
                if (histogram[i] > 0)
                {
                    if (minValue == -1)
                    {
                        minValue = i;
                    }
                    maxValue = i;
                }
            }
            Min = minValue != -1 ? minValue : (int?)null;
            Max = maxValue != -1 ? maxValue : (int?)null;

            // Range (범위)
            if (Min.HasValue && Max.HasValue)
            {
                Range = Max.Value - Min.Value;
            }
            else
            {
                Range = null;
            }
        }
    }
}
