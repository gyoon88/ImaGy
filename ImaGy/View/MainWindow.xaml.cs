using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ImaGy.ViewModel;

namespace ImaGy.View
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private ScaleTransform _beforeScaleTransform;
        private ScaleTransform _afterScaleTransform;
        private Point _lastMousePosition;
        private bool _isPanning;
        private MainViewModel _viewModel;

        public Action<BitmapSource>? OnImageLoadedCallback;

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _beforeScaleTransform = (ScaleTransform)BeforeImageControl.LayoutTransform;
            _afterScaleTransform = (ScaleTransform)AfterImageControl.LayoutTransform;
            _viewModel = (MainViewModel)this.DataContext;

            // Set the callback for the ViewModel
            _viewModel.ImageLoadedCallback = ResetImageZoom;

            // Initial update of ZoomLevel and fit image
            if (_viewModel.BeforeImage != null)
            {
                ResetImageZoom(_viewModel.BeforeImage);
            }
            else
            {
                _viewModel.UpdateZoomLevel(_beforeScaleTransform.ScaleX);
            }
        }

        public void ResetImageZoom(BitmapSource imageSource)
        {
            if (imageSource == null) return;

            // Calculate scale for BeforeImageControl
            double scaleXBefore = BeforeImageScrollViewer.ActualWidth / imageSource.PixelWidth;
            double scaleYBefore = BeforeImageScrollViewer.ActualHeight / imageSource.PixelHeight;
            double initialScaleBefore = Math.Min(scaleXBefore, scaleYBefore);

            _beforeScaleTransform.ScaleX = initialScaleBefore;
            _beforeScaleTransform.ScaleY = initialScaleBefore;

            // Calculate scale for AfterImageControl (assuming it will display the same image initially)
            double scaleXAfter = AfterImageScrollViewer.ActualWidth / imageSource.PixelWidth;
            double scaleYAfter = AfterImageScrollViewer.ActualHeight / imageSource.PixelHeight;
            double initialScaleAfter = Math.Min(scaleXAfter, scaleYAfter);

            _afterScaleTransform.ScaleX = initialScaleAfter;
            _afterScaleTransform.ScaleY = initialScaleAfter;

            // Reset scroll positions
            BeforeImageScrollViewer.ScrollToHome();
            AfterImageScrollViewer.ScrollToHome();

            // Update ViewModel's ZoomLevel
            _viewModel.UpdateZoomLevel(initialScaleBefore);
        }

        private void ImageControl_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            e.Handled = true; // Consume the event to prevent ScrollViewer from handling it

            Image imageControl = (Image)sender;
            ScrollViewer scrollViewer = (ScrollViewer)imageControl.Parent; // Get parent ScrollViewer
            ScaleTransform scaleTransform = (imageControl == BeforeImageControl) ? _beforeScaleTransform : _afterScaleTransform;

            double zoomFactor = 1.1; // Zoom in/out factor

            if (e.Delta > 0)
            {
                // Zoom in
                scaleTransform.ScaleX *= zoomFactor;
                scaleTransform.ScaleY *= zoomFactor;
            }
            else
            {
                // Zoom out
                scaleTransform.ScaleX /= zoomFactor;
                scaleTransform.ScaleY /= zoomFactor;
            }

            // Update ViewModel
            _viewModel.UpdateZoomLevel(scaleTransform.ScaleX);

            // Adjust scroll position to keep the mouse pointer centered (optional, but good UX)
            Point mousePos = e.GetPosition(imageControl);
            double centerPointX = mousePos.X / imageControl.ActualWidth;
            double centerPointY = mousePos.Y / imageControl.ActualHeight;

            scrollViewer.ScrollToHorizontalOffset(centerPointX * scrollViewer.ScrollableWidth);
            scrollViewer.ScrollToVerticalOffset(centerPointY * scrollViewer.ScrollableHeight);
        }

        private void ImageScrollViewer_MouseDown(object sender, MouseButtonEventArgs e)
        {
            ScrollViewer scrollViewer = (ScrollViewer)sender;
            if (e.ChangedButton == MouseButton.Left)
            {
                _isPanning = true;
                _lastMousePosition = e.GetPosition(scrollViewer);
                scrollViewer.CaptureMouse();
            }
        }

        private void ImageScrollViewer_MouseMove(object sender, MouseEventArgs e)
        {
            ScrollViewer scrollViewer = (ScrollViewer)sender;
            if (_isPanning)
            {
                Point currentMousePosition = e.GetPosition(scrollViewer);
                double deltaX = currentMousePosition.X - _lastMousePosition.X;
                double deltaY = currentMousePosition.Y - _lastMousePosition.Y;

                scrollViewer.ScrollToHorizontalOffset(scrollViewer.HorizontalOffset - deltaX);
                scrollViewer.ScrollToVerticalOffset(scrollViewer.VerticalOffset - deltaY);

                _lastMousePosition = currentMousePosition;
            }
        }

        private void ImageScrollViewer_MouseUp(object sender, MouseButtonEventArgs e)
        {
            ScrollViewer scrollViewer = (ScrollViewer)sender;
            if (e.ChangedButton == MouseButton.Left)
            {
                _isPanning = false;
                scrollViewer.ReleaseMouseCapture();
            }
        }
    }
}