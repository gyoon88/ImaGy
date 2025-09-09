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
                DrawHistogram(); // �ʱ� ������׷� �׸���
            }

            private void ViewModel_PropertyChanged(object sender, PropertyChangedEventArgs e)
            {
                // ViewModel�� �ֿ� ������ �Ӽ��� ����Ǹ� �׷����� �ٽ� �׸����� ��ȣ�� �����ϴ�.
                // OnPropertyChanged�� ���� �� ȣ��ǹǷ�, ��� ����� ���� �� ������Ʈ�Ǵ� MaxHistogramValue�� �������� ��� ���� ȿ�����Դϴ�.
                if (e.PropertyName == nameof(HistogramViewModel.MaxHistogramValue))
                {
                    DrawHistogram();
                }
            }

            private void HistogramCanvas_SizeChanged(object sender, SizeChangedEventArgs e)
            {
                // â ũ�Ⱑ ����� ���� �׷����� �ٽ� �׸��ϴ�.
                DrawHistogram();
            }

            private void DrawHistogram()
            {
                HistogramCanvas.Children.Clear();
                if (_viewModel == null) return;

                // ViewModel�� �÷��׿� ���� �׸��� ������ �б��մϴ�.
                if (_viewModel.IsColorImage)
                {
                    // �������� R, G, B �������� �� ä���� ���ļ� �׸��ϴ�.
                    DrawChannelHistogram(_viewModel.R_HistogramData, Color.FromArgb(128, 255, 0, 0)); // Red
                    DrawChannelHistogram(_viewModel.G_HistogramData, Color.FromArgb(128, 0, 255, 0)); // Green
                    DrawChannelHistogram(_viewModel.B_HistogramData, Color.FromArgb(128, 0, 0, 255)); // Blue
                }
                else
                {
                    // ��� ������׷��� �׸��ϴ�.
                    DrawChannelHistogram(_viewModel.GrayscaleHistogramData, Colors.WhiteSmoke);
                }
            }

            // �� ä���� �׸��� ������ ���� �޼���� �и��Ͽ� �ڵ� �ߺ��� ���Դϴ�.
            private void DrawChannelHistogram(int[]? data, Color color)
            {
                if (data == null || data.Length == 0) return;

                int max = _viewModel.MaxHistogramValue;
                if (max == 0) return;

                double canvasWidth = HistogramCanvas.ActualWidth;
                double canvasHeight = HistogramCanvas.ActualHeight;
                double barWidth = canvasWidth / data.Length;

                var brush = new SolidColorBrush(color);
                brush.Freeze(); // ���� ����� ���� WPF ���ҽ��� �����մϴ�.

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

