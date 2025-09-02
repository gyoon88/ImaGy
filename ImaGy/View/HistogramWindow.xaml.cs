using ImaGy.ViewModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ImaGy.View
{
    /// <summary>
    /// Interaction logic for HistogramWindow.xaml
    /// </summary>
    public partial class HistogramWindow : Window
    {
        public HistogramWindow()
        {
            InitializeComponent();
        }

        private void HistogramCanvas_Loaded(object sender, RoutedEventArgs e)
        {
            RedrawHistogram();
        }

        private void HistogramCanvas_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            RedrawHistogram();
        }

        public void RedrawHistogram()
        {
            if (DataContext is HistogramViewModel viewModel)
            {
                DrawHistogram(viewModel.HistogramData, viewModel.MaxHistogramValue);
            }
        }

        private void DrawHistogram(int[]? histogramData, int maxVal)
        {
            HistogramCanvas.Children.Clear();
            if (histogramData == null || histogramData.Length == 0 || maxVal == 0) return;

            double canvasWidth = HistogramCanvas.ActualWidth;
            double canvasHeight = HistogramCanvas.ActualHeight;

            if (canvasWidth == 0 || canvasHeight == 0) return; // Avoid division by zero if not rendered yet

            double barWidth = canvasWidth / histogramData.Length;

            for (int i = 0; i < histogramData.Length; i++)
            {
                double barHeight = (double)histogramData[i] / maxVal * canvasHeight;

                Rectangle rect = new Rectangle
                {
                    Width = barWidth,
                    Height = barHeight,
                    Fill = Brushes.Blue
                };

                Canvas.SetLeft(rect, i * barWidth);
                Canvas.SetBottom(rect, 0); // Align bars to the bottom

                HistogramCanvas.Children.Add(rect);
            }
        }
    }
}
