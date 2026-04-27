using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Collections.Generic;
using System.ComponentModel;
using ImaGy.Grids;
using ImaGy.Models;

namespace ImaGy.ViewModels
{
    public class HistogramViewModel : BaseViewModel
    {
        private MainViewModel? _mainViewModel;
        private const int DefaultBinCount = 256;

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

        private bool isFloatHistogram;
        public bool IsFloatHistogram
        {
            get => isFloatHistogram;
            private set => SetProperty(ref isFloatHistogram, value);
        }

        private double histogramValueMin;
        public double HistogramValueMin
        {
            get => histogramValueMin;
            private set => SetProperty(ref histogramValueMin, value);
        }

        private double histogramValueMax = 255;
        public double HistogramValueMax
        {
            get => histogramValueMax;
            private set => SetProperty(ref histogramValueMax, value);
        }

        public string XAxisTitle => IsFloatHistogram ? "Value" : "Pixel level";

        #endregion

        #region Public Properties for Statistics
        // 요약 통계량
        private double? mean;
        private double? std;
        private double? median;
        private double? mode;
        private double? max;
        private double? min;
        private double? range;

        public double? Mean { get => mean; private set => SetProperty(ref mean, value); }
        public double? Std { get => std; private set => SetProperty(ref std, value); }
        public double? Median { get => median; private set => SetProperty(ref median, value); }
        public double? Mode { get => mode; private set => SetProperty(ref mode, value); }
        public double? Max { get => max; private set => SetProperty(ref max, value); }
        public double? Min { get => min; private set => SetProperty(ref min, value); }
        public double? Range { get => range; private set => SetProperty(ref range, value); }
        #endregion

        public HistogramViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _mainViewModel.PropertyChanged += MainViewModel_PropertyChanged;
            UpdateAllHistograms();
        }

        /// <summary>CSV 격자 Diff 등 부동소수 그리드용. 기존 HistogramWindow UI(256 bin) 재사용.</summary>
        public HistogramViewModel(FloatGrid grid, bool[]? mask)
        {
            _mainViewModel = null;
            LoadHistogramFromFloatGrid(grid, mask);
        }

        private void LoadHistogramFromFloatGrid(FloatGrid grid, bool[]? mask)
        {
            var stats = GridStatisticsService.Compute(grid, mask, null, null);
            var hist = GridStatisticsService.ComputeHistogram(grid, mask, null, null, DefaultBinCount);
            IsColorImage = false;
            IsFloatHistogram = true;
            GrayscaleHistogramData = hist;
            R_HistogramData = G_HistogramData = B_HistogramData = null;
            MaxHistogramValue = hist.Length > 0 ? hist.Max() : 0;
            if (stats.Count > 0)
            {
                HistogramValueMin = stats.Min;
                HistogramValueMax = stats.Max;
            }
            else
            {
                HistogramValueMin = 0;
                HistogramValueMax = 1;
            }
            SetFloatStatistics(stats, hist);
            OnPropertyChanged(nameof(IsColorImage));
            OnPropertyChanged(nameof(IsFloatHistogram));
            OnPropertyChanged(nameof(HistogramValueMin));
            OnPropertyChanged(nameof(HistogramValueMax));
            OnPropertyChanged(nameof(XAxisTitle));
            OnPropertyChanged(nameof(R_HistogramData));
            OnPropertyChanged(nameof(G_HistogramData));
            OnPropertyChanged(nameof(B_HistogramData));
            OnPropertyChanged(nameof(GrayscaleHistogramData));
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
            if (_mainViewModel == null) return;

            BitmapSource? imageSource = _mainViewModel.AfterImage ?? _mainViewModel.BeforeImage;
            if (imageSource != null)
            {
                if (imageSource.Format == PixelFormats.Gray8)
                {
                    IsColorImage = false;
                    IsFloatHistogram = false;
                    GrayscaleHistogramData = ServeHistogram.CalculateGrayscaleHistogram(imageSource);
                    R_HistogramData = G_HistogramData = B_HistogramData = null;
                    MaxHistogramValue = GrayscaleHistogramData.Any() ? GrayscaleHistogramData.Max() : 0;
                    HistogramValueMin = 0;
                    HistogramValueMax = 255;


                    CalculateAndSetStatistics(GrayscaleHistogramData);
                }
                else
                {
                    IsColorImage = true;
                    IsFloatHistogram = false;
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
                    HistogramValueMin = 0;
                    HistogramValueMax = 255;

                    // 컬러 이미지의 경우, 그레이스케일로 변환하여 통계 계산
                    var statsHistogram = ServeHistogram.CalculateGrayscaleHistogram(imageSource);
                    CalculateAndSetStatistics(statsHistogram);
                }
            }
            else
            {
                // 이미지가 없을 경우 모든 데이터 초기화
                IsColorImage = false;
                IsFloatHistogram = false;
                GrayscaleHistogramData = R_HistogramData = G_HistogramData = B_HistogramData = null;
                MaxHistogramValue = 0;
                HistogramValueMin = 0;
                HistogramValueMax = 255;
                ClearStatistics();
            }

            // View가 데이터를 다시 그리도록 모든 관련 속성의 변경을 알림.
            OnPropertyChanged(nameof(IsColorImage));
            OnPropertyChanged(nameof(IsFloatHistogram));
            OnPropertyChanged(nameof(HistogramValueMin));
            OnPropertyChanged(nameof(HistogramValueMax));
            OnPropertyChanged(nameof(XAxisTitle));
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
        /// 주어진 히스토그램 데이터를 기반으로 요약 통계량을 계산하고 속성을 설정.
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
            Min = minValue != -1 ? (double?)minValue : null;
            Max = maxValue != -1 ? (double?)maxValue : null;

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

        private void SetFloatStatistics(GridStatisticsResult stats, int[] histogram)
        {
            if (stats.Count <= 0)
            {
                ClearStatistics();
                return;
            }

            Mean = stats.Mean;
            Std = stats.Std;
            Median = stats.Median;
            Min = stats.Min;
            Max = stats.Max;
            Range = stats.Max - stats.Min;

            int modeBin = 0;
            int bestCount = 0;
            for (int i = 0; i < histogram.Length; i++)
            {
                if (histogram[i] > bestCount)
                {
                    bestCount = histogram[i];
                    modeBin = i;
                }
            }
            Mode = GetBinCenter(modeBin, histogram.Length);
        }

        public double GetValueAtFraction(double fraction)
        {
            double t = Math.Clamp(fraction, 0, 1);
            return HistogramValueMin + (HistogramValueMax - HistogramValueMin) * t;
        }

        public int GetBinCount => GrayscaleHistogramData?.Length ?? 256;

        public double GetBinStart(int binIndex, int binCount)
        {
            if (binCount <= 0) return HistogramValueMin;
            if (HistogramValueMax <= HistogramValueMin) return HistogramValueMin;
            double width = (HistogramValueMax - HistogramValueMin) / binCount;
            return HistogramValueMin + Math.Clamp(binIndex, 0, binCount - 1) * width;
        }

        public double GetBinEnd(int binIndex, int binCount)
        {
            if (binCount <= 0) return HistogramValueMax;
            if (HistogramValueMax <= HistogramValueMin) return HistogramValueMax;
            double width = (HistogramValueMax - HistogramValueMin) / binCount;
            return HistogramValueMin + Math.Clamp(binIndex + 1, 1, binCount) * width;
        }

        public double GetBinCenter(int binIndex, int binCount)
        {
            if (HistogramValueMax <= HistogramValueMin) return HistogramValueMin;
            return 0.5 * (GetBinStart(binIndex, binCount) + GetBinEnd(binIndex, binCount));
        }
    }
}
