using ImaGy.ViewModels;
using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ImaGy.View
{
    public partial class HistogramWindow : Window
    {
        private HistogramViewModel? _viewModel;
        private bool _useLogScale;

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
            DrawUI();
        }

        private void ViewModel_PropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(HistogramViewModel.MaxHistogramValue)
                || e.PropertyName == nameof(HistogramViewModel.HistogramValueMin)
                || e.PropertyName == nameof(HistogramViewModel.HistogramValueMax)
                || e.PropertyName == nameof(HistogramViewModel.IsFloatHistogram)
                || e.PropertyName == nameof(HistogramViewModel.SampleModeIndex)
                || e.PropertyName == nameof(HistogramViewModel.GrayscaleHistogramData)
                || e.PropertyName == nameof(HistogramViewModel.R_HistogramData))
            {
                DrawUI();
            }
        }

        private void OpenLineProfile_Click(object sender, RoutedEventArgs e)
        {
            if (DataContext is not HistogramViewModel vm)
                return;
            if (!vm.TryOpenLineProfile(out var lpVm))
            {
                System.Windows.MessageBox.Show(this, "라인 프로파일을 열 수 없습니다. 메인에 이미지가 있는지 확인하세요.", "ImaGy",
                    MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }
            new LineProfileWindow { DataContext = lpVm }.Show();
        }

        private void HistogramCanvas_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            DrawUI();
        }

        private void UseLogScaleCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            _useLogScale = GetUseLogScaleCheckBox()?.IsChecked == true;
            DrawUI();
        }

        private void DrawUI()
        {
            HistogramCanvas.Children.Clear();
            XAxisLabelsPanel.Children.Clear();
            YAxisLabelsPanel.Children.Clear();

            if (_viewModel == null || _viewModel.MaxHistogramValue == 0) return;

            DrawAxes();
            DrawHistogram();
            DrawXAxisLabels();
            DrawYAxisLabels();
        }

        private void DrawAxes()
        {
            double canvasWidth = HistogramCanvas.ActualWidth;
            double canvasHeight = HistogramCanvas.ActualHeight;

            // Y-Axis Line
            var yAxis = new Line
            {
                X1 = 0, Y1 = 0,
                X2 = 0, Y2 = canvasHeight,
                Stroke = System.Windows.Media.Brushes.WhiteSmoke, StrokeThickness = 1
            };
            HistogramCanvas.Children.Add(yAxis);

            // X-Axis Line
            var xAxis = new Line
            {
                X1 = 0, Y1 = canvasHeight,
                X2 = canvasWidth, Y2 = canvasHeight,
                Stroke = System.Windows.Media.Brushes.WhiteSmoke, StrokeThickness = 1
            };
            HistogramCanvas.Children.Add(xAxis);
        }

        private void DrawHistogram()
        {
            if (_viewModel == null) return;

            if (_viewModel.IsColorImage)
            {
                DrawChannelHistogram(_viewModel.R_HistogramData, System.Windows.Media.Color.FromArgb(128, 255, 0, 0)); // Red
                DrawChannelHistogram(_viewModel.G_HistogramData, System.Windows.Media.Color.FromArgb(128, 0, 255, 0)); // Green
                DrawChannelHistogram(_viewModel.B_HistogramData, System.Windows.Media.Color.FromArgb(128, 0, 0, 255)); // Blue
            }
            else
            {
                DrawChannelHistogram(_viewModel.GrayscaleHistogramData, Colors.WhiteSmoke);
            }
        }

        private void DrawChannelHistogram(int[]? data, System.Windows.Media.Color color)
        {
            if (data == null || data.Length == 0 || _viewModel == null) return;

            int max = _viewModel.MaxHistogramValue;
            if (max == 0) return;

            double canvasWidth = HistogramCanvas.ActualWidth;
            double canvasHeight = HistogramCanvas.ActualHeight;
            double barWidth = canvasWidth / data.Length;
            double logDenom = Math.Log10(max + 1.0);

            var brush = new SolidColorBrush(color);
            brush.Freeze();

            for (int i = 0; i < data.Length; i++)
            {
                double normalized = _useLogScale
                    ? (logDenom <= 0 ? 0 : Math.Log10(data[i] + 1.0) / logDenom)
                    : (double)data[i] / max;
                double barHeight = normalized * canvasHeight;
                if (barHeight <= 0) continue;

                var bar = new System.Windows.Shapes.Rectangle
                {
                    Width = barWidth,
                    Height = barHeight,
                    Fill = brush,
                    ToolTip = BuildBinTooltip(i, data[i], data.Length)
                };

                Canvas.SetLeft(bar, i * barWidth);
                Canvas.SetBottom(bar, 0);
                HistogramCanvas.Children.Add(bar);
            }
        }

        private void DrawXAxisLabels()
        {
            if (_viewModel == null) return;
            double canvasWidth = HistogramCanvas.ActualWidth;
            const int labelCount = 7;
            if (canvasWidth <= 0) return;

            for (int i = 0; i < labelCount; i++)
            {
                double fraction = (double)i / (labelCount - 1);
                double value = _viewModel.GetValueAtFraction(fraction);
                var label = new TextBlock
                {
                    Text = FormatXValue(value),
                    Foreground = System.Windows.Media.Brushes.WhiteSmoke,
                    HorizontalAlignment = System.Windows.HorizontalAlignment.Center
                };
                label.Measure(new System.Windows.Size(double.PositiveInfinity, double.PositiveInfinity));
                double x = fraction * canvasWidth;
                Canvas.SetLeft(label, Math.Max(0, Math.Min(canvasWidth - label.DesiredSize.Width, x - label.DesiredSize.Width / 2)));
                Canvas.SetTop(label, 0);
                XAxisLabelsPanel.Children.Add(label);
            }
        }

        private void DrawYAxisLabels()
        {
            if (_viewModel == null) return;

            double panelHeight = Math.Max(1, HistogramCanvas.ActualHeight);
            const int labelCount = 5;
            int max = _viewModel.MaxHistogramValue;
            YAxisLabelsPanel.Children.Clear();
            double logDenom = Math.Log10(max + 1.0);

            for (int i = 0; i < labelCount; i++)
            {
                double valueFraction = (double)i / (labelCount - 1);
                int value = _useLogScale
                    ? (int)Math.Round(Math.Pow(10.0, (1.0 - valueFraction) * logDenom) - 1.0)
                    : (int)Math.Round(max * (1 - valueFraction));
                double y = valueFraction * panelHeight;

                var label = new TextBlock
                {
                    Text = FormatYAxisLabel(value),
                    Foreground = System.Windows.Media.Brushes.WhiteSmoke,
                    VerticalAlignment = VerticalAlignment.Center
                };
                label.Measure(new System.Windows.Size(double.PositiveInfinity, double.PositiveInfinity));
                Canvas.SetLeft(label, 0);
                Canvas.SetTop(label, Math.Max(0, y - label.DesiredSize.Height / 2));
                YAxisLabelsPanel.Children.Add(label);
            }
        }

        private string BuildBinTooltip(int binIndex, int count, int binCount)
        {
            if (_viewModel == null)
                return $"bin={binIndex}, count={count}";
            double start = _viewModel.GetBinStart(binIndex, binCount);
            double end = _viewModel.GetBinEnd(binIndex, binCount);
            double center = _viewModel.GetBinCenter(binIndex, binCount);
            if (_viewModel.IsFloatHistogram)
                return $"x=[{FormatValue(start)}, {FormatValue(end)})\ncenter={FormatValue(center)}\ncount={count}";
            return $"pixel=[{start:F0}, {Math.Max(start, end - 1):F0}]\ncount={count}";
        }

        private string FormatXValue(double value)
        {
            if (_viewModel?.IsFloatHistogram == true) return FormatValue(value);
            return value.ToString("F0");
        }

        private static string FormatValue(double value)
        {
            double abs = Math.Abs(value);
            if ((abs > 0 && abs < 1e-3) || abs >= 1e5) return value.ToString("0.###E+0");
            return value.ToString("F4");
        }

        private string FormatYAxisLabel(int value)
        {
            if (value >= 1000000) return $"{(double)value / 1000000:0.#}M";
            if (value >= 1000) return $"{(double)value / 1000:0.#}K";
            return value.ToString();
        }

        private void HistogramCanvas_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (_viewModel == null) return;
            int binCount = _viewModel.IsColorImage
                ? Math.Max(_viewModel.R_HistogramData?.Length ?? 0, Math.Max(_viewModel.G_HistogramData?.Length ?? 0, _viewModel.B_HistogramData?.Length ?? 0))
                : (_viewModel.GrayscaleHistogramData?.Length ?? 0);
            if (binCount <= 0 || HistogramCanvas.ActualWidth <= 0) return;

            double x = e.GetPosition(HistogramCanvas).X;
            int bin = (int)Math.Floor(x / HistogramCanvas.ActualWidth * binCount);
            bin = Math.Clamp(bin, 0, binCount - 1);

            double start = _viewModel.GetBinStart(bin, binCount);
            double end = _viewModel.GetBinEnd(bin, binCount);
            double center = _viewModel.GetBinCenter(bin, binCount);

            if (_viewModel.IsColorImage)
            {
                int r = GetCount(_viewModel.R_HistogramData, bin);
                int g = GetCount(_viewModel.G_HistogramData, bin);
                int b = GetCount(_viewModel.B_HistogramData, bin);
                SetHoverInfoText($"bin {bin} x={start:F0}~{Math.Max(start, end - 1):F0} | R={r}, G={g}, B={b}");
            }
            else if (_viewModel.IsFloatHistogram)
            {
                int c = GetCount(_viewModel.GrayscaleHistogramData, bin);
                SetHoverInfoText($"bin {bin} x=[{FormatValue(start)}, {FormatValue(end)}) center={FormatValue(center)} count={c}");
            }
            else
            {
                int c = GetCount(_viewModel.GrayscaleHistogramData, bin);
                SetHoverInfoText($"bin {bin} pixel={start:F0}~{Math.Max(start, end - 1):F0} count={c}");
            }
        }

        private void HistogramCanvas_MouseLeave(object sender, System.Windows.Input.MouseEventArgs e)
        {
            SetHoverInfoText("마우스를 올려 bin 값 확인");
        }

        private static int GetCount(int[]? data, int index)
        {
            if (data == null || index < 0 || index >= data.Length) return 0;
            return data[index];
        }

        private void SetHoverInfoText(string text)
        {
            if (FindName("HoverInfoText") is TextBlock tb)
                tb.Text = text;
        }

        private System.Windows.Controls.CheckBox? GetUseLogScaleCheckBox() => FindName("UseLogScaleCheckBox") as System.Windows.Controls.CheckBox;
    }
}