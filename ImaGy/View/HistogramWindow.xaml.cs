using ImaGy.ViewModels;
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
            YAxisLabelsPanel.Children.Clear();
            XAxisLabelsPanel.Children.Clear();

            if (histogramData == null || histogramData.Length == 0 || maxVal == 0) return;

            double canvasWidth = HistogramCanvas.ActualWidth;
            double canvasHeight = HistogramCanvas.ActualHeight;

            if (canvasWidth == 0 || canvasHeight == 0) return; // Avoid division by zero if not rendered yet

            DrawGridAndLabels(canvasWidth, canvasHeight, maxVal);

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

        private void DrawGridAndLabels(double canvasWidth, double canvasHeight, int maxVal)
        {
            // Draw Horizontal Grid Lines and Y-axis Labels
            int horizontalLines = 5; // 0%, 25%, 50%, 75%, 100%
            YAxisLabelsPanel.Children.Clear();
            YAxisLabelsPanel.RowDefinitions.Clear(); // Clear existing row definitions

            for (int i = 0; i < horizontalLines; i++)
            {
                // Add RowDefinition for each label
                YAxisLabelsPanel.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });

                double y = canvasHeight - (canvasHeight / (horizontalLines - 1)) * i;
                if (i == 0) y = canvasHeight; // Ensure bottom line is exactly at the bottom

                Line horizontalLine = new Line
                {
                    X1 = 0, Y1 = y,
                    X2 = canvasWidth, Y2 = y,
                    Stroke = Brushes.LightGray,
                    StrokeDashArray = new DoubleCollection { 2, 2 }
                };
                HistogramCanvas.Children.Add(horizontalLine);

                TextBlock yLabel = new TextBlock
                {
                    Text = (maxVal / (horizontalLines - 1) * i).ToString(),
                    VerticalAlignment = VerticalAlignment.Center,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Margin = new Thickness(0, 0, 5, 0),
                    Foreground = Brushes.White
                };
                Grid.SetRow(yLabel, horizontalLines - 1 - i); // Set row for the label (reverse order)
                YAxisLabelsPanel.Children.Add(yLabel);
            }

            // Draw Vertical Grid Lines and X-axis Labels
            int verticalLines = 5; // 0, 64, 128, 192, 255
            for (int i = 0; i < verticalLines; i++)
            {
                double x = (canvasWidth / (verticalLines - 1)) * i;

                Line verticalLine = new Line
                {
                    X1 = x, Y1 = 0,
                    X2 = x, Y2 = canvasHeight,
                    Stroke = Brushes.LightGray,
                    StrokeDashArray = new DoubleCollection { 2, 2 }
                };
                HistogramCanvas.Children.Add(verticalLine);

                TextBlock xLabel = new TextBlock
                {
                    Text = (255 / (verticalLines - 1) * i).ToString(),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Top,
                    Width = canvasWidth / (verticalLines - 1), // Distribute width
                    Foreground = Brushes.White
                };
                XAxisLabelsPanel.Children.Add(xLabel);
            }
        }
    }
}
