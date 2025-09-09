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

        // View가 직접 바인딩할 데이터 속성들
        public int[]? R_HistogramData { get; private set; }
        public int[]? G_HistogramData { get; private set; }
        public int[]? B_HistogramData { get; private set; }
        public int[]? GrayscaleHistogramData { get; private set; }

        // 모든 채널 중 가장 큰 값을 저장하여 Y축 스케일을 통일합니다.
        private int maxHistogramValue;
        public int MaxHistogramValue
        {
            get => maxHistogramValue;
            private set => SetProperty(ref maxHistogramValue, value);
        }

        // View에게 컬러/흑백 중 무엇을 그려야 할지 알려주는 플래그
        private bool isColorImage;
        public bool IsColorImage
        {
            get => isColorImage;
            private set => SetProperty(ref isColorImage, value);
        }

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
                    R_HistogramData = G_HistogramData = B_HistogramData = null; // 컬러 데이터 초기화
                    MaxHistogramValue = GrayscaleHistogramData.Any() ? GrayscaleHistogramData.Max() : 0;
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
                    GrayscaleHistogramData = null; // 흑백 데이터 초기화

                    // 세 채널 중 가장 큰 값을 찾아 MaxValue로 설정
                    int maxR = R_HistogramData?.Max() ?? 0;
                    int maxG = G_HistogramData?.Max() ?? 0;
                    int maxB = B_HistogramData?.Max() ?? 0;
                    MaxHistogramValue = Math.Max(maxR, Math.Max(maxG, maxB));
                }
            }
            else
            {
                // 이미지가 없을 경우 모든 데이터 초기화
                IsColorImage = false;
                GrayscaleHistogramData = R_HistogramData = G_HistogramData = B_HistogramData = null;
                MaxHistogramValue = 0;
            }

            // View가 데이터를 다시 그리도록 모든 관련 속성의 변경을 알립니다.
            OnPropertyChanged(nameof(IsColorImage));
            OnPropertyChanged(nameof(R_HistogramData));
            OnPropertyChanged(nameof(G_HistogramData));
            OnPropertyChanged(nameof(B_HistogramData));
            OnPropertyChanged(nameof(GrayscaleHistogramData));
        }
    }
}

