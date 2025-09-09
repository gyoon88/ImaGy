using ImaGy.ViewModels;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ImaGy.View
{
        public partial class HistogramWindow : Window
        {
            private HistogramViewModel _viewModel;

            public HistogramWindow()
            {
                InitializeComponent();
                Loaded += HistogramWindow_Loaded;
            }

            private void HistogramWindow_Loaded(object sender, RoutedEventArgs e)
            {
                _viewModel = DataContext as HistogramViewModel;
                if (_viewModel == null) return;

                _viewModel.PropertyChanged += ViewModel_PropertyChanged;
                DrawHistogram(); // 초기 히스토그램 그리기
            }

            private void ViewModel_PropertyChanged(object sender, PropertyChangedEventArgs e)
            {
                // ViewModel의 주요 데이터 속성이 변경되면 그래프를 다시 그리도록 신호를 보냅니다.
                // OnPropertyChanged가 여러 번 호출되므로, 모든 계산이 끝난 후 업데이트되는 MaxHistogramValue를 기준으로 삼는 것이 효율적입니다.
                if (e.PropertyName == nameof(HistogramViewModel.MaxHistogramValue))
                {
                    DrawHistogram();
                }
            }

            private void HistogramCanvas_SizeChanged(object sender, SizeChangedEventArgs e)
            {
                // 창 크기가 변경될 때도 그래프를 다시 그립니다.
                DrawHistogram();
            }

            private void DrawHistogram()
            {
                HistogramCanvas.Children.Clear();
                if (_viewModel == null) return;

                // ViewModel의 플래그에 따라 그리기 로직을 분기합니다.
                if (_viewModel.IsColorImage)
                {
                    // 반투명한 R, G, B 색상으로 각 채널을 겹쳐서 그립니다.
                    DrawChannelHistogram(_viewModel.R_HistogramData, Color.FromArgb(128, 255, 0, 0)); // Red
                    DrawChannelHistogram(_viewModel.G_HistogramData, Color.FromArgb(128, 0, 255, 0)); // Green
                    DrawChannelHistogram(_viewModel.B_HistogramData, Color.FromArgb(128, 0, 0, 255)); // Blue
                }
                else
                {
                    // 흑백 히스토그램을 그립니다.
                    DrawChannelHistogram(_viewModel.GrayscaleHistogramData, Colors.WhiteSmoke);
                }
            }

            // 각 채널을 그리는 로직을 별도 메서드로 분리하여 코드 중복을 줄입니다.
            private void DrawChannelHistogram(int[]? data, Color color)
            {
                if (data == null || data.Length == 0) return;

                int max = _viewModel.MaxHistogramValue;
                if (max == 0) return;

                double canvasWidth = HistogramCanvas.ActualWidth;
                double canvasHeight = HistogramCanvas.ActualHeight;
                double barWidth = canvasWidth / data.Length;

                var brush = new SolidColorBrush(color);
                brush.Freeze(); // 성능 향상을 위해 WPF 리소스를 고정합니다.

                for (int i = 0; i < data.Length; i++)
                {
                    double barHeight = (double)data[i] / max * canvasHeight;
                    if (barHeight <= 0) continue;

                    var bar = new Rectangle
                    {
                        Width = barWidth,
                        Height = barHeight,
                        Fill = brush
                    };

                    Canvas.SetLeft(bar, i * barWidth);
                    Canvas.SetBottom(bar, 0);
                    HistogramCanvas.Children.Add(bar);
                }
            }
        }
    }

