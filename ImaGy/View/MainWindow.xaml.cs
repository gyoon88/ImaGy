using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ImaGy.ViewModels;

namespace ImaGy.View
{
    public partial class MainWindow : Window
    {
        private ScaleTransform beforeScaleTransform;
        private ScaleTransform afterScaleTransform;
        private Point lastMousePosition;
        private bool isPanning;
        private MainViewModel viewModel;

        public Action<BitmapSource>? OnImageLoadedCallback;

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            beforeScaleTransform = (ScaleTransform)BeforeImageControl.LayoutTransform;
            afterScaleTransform = (ScaleTransform)AfterImageControl.LayoutTransform;
            viewModel = (MainViewModel)this.DataContext;

            // Set the callback for the ViewModel
            viewModel.ImageLoadedCallback = ResetImageZoom;
            viewModel.beforeScaleTransform = beforeScaleTransform; // Pass the ScaleTransform

            // Initial update of ZoomLevel and fit image
            if (viewModel.BeforeImage != null)
            {
                ResetImageZoom(viewModel.BeforeImage);
            }
            else
            {
                viewModel.UpdateZoomLevel(beforeScaleTransform.ScaleX);
            }
        }

        public void ResetImageZoom(BitmapSource imageSource)
        {
            if (imageSource == null) return;

            // Calculate scale for BeforeImageControl
            double scaleXBefore = BeforeImageScrollViewer.ActualWidth / imageSource.PixelWidth;
            double scaleYBefore = BeforeImageScrollViewer.ActualHeight / imageSource.PixelHeight;
            double initialScaleBefore = Math.Min(scaleXBefore, scaleYBefore);

            beforeScaleTransform.ScaleX = initialScaleBefore;
            beforeScaleTransform.ScaleY = initialScaleBefore;

            // Calculate scale for AfterImageControl (assuming it will display the same image initially)
            double scaleXAfter = AfterImageScrollViewer.ActualWidth / imageSource.PixelWidth;
            double scaleYAfter = AfterImageScrollViewer.ActualHeight / imageSource.PixelHeight;
            double initialScaleAfter = Math.Min(scaleXAfter, scaleYAfter);

            afterScaleTransform.ScaleX = initialScaleAfter;
            afterScaleTransform.ScaleY = initialScaleAfter;

            // Reset scroll positions
            BeforeImageScrollViewer.ScrollToHome();
            AfterImageScrollViewer.ScrollToHome();

            // Update ViewModel's ZoomLevel
            viewModel.UpdateZoomLevel(initialScaleBefore);
        }

        private void BeforeImageScrollViewer_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            if (DataContext is MainViewModel viewModel)
            {
                viewModel.ScrollViewerHorizontalOffset = BeforeImageScrollViewer.HorizontalOffset;
                viewModel.ScrollViewerVerticalOffset = BeforeImageScrollViewer.VerticalOffset;
                viewModel.ScrollViewerViewportWidth = BeforeImageScrollViewer.ViewportWidth;
                viewModel.ScrollViewerViewportHeight = BeforeImageScrollViewer.ViewportHeight;
            }
        }

        private void AfterImageScrollViewer_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            if (DataContext is MainViewModel viewModel)
            {
                viewModel.ScrollViewerHorizontalOffset = AfterImageScrollViewer.HorizontalOffset;
                viewModel.ScrollViewerVerticalOffset = AfterImageScrollViewer.VerticalOffset;
                viewModel.ScrollViewerViewportWidth = AfterImageScrollViewer.ViewportWidth;
                viewModel.ScrollViewerViewportHeight = AfterImageScrollViewer.ViewportHeight;
            }
        }

        private void ImageControl_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            e.Handled = true; // Consume the event to prevent ScrollViewer from handling it

            Image imageControl = (Image)sender;
            ScrollViewer scrollViewer = FindAncestor<ScrollViewer>(imageControl); // Safely get parent ScrollViewer
            if (scrollViewer == null) return;
            ScaleTransform scaleTransform = (imageControl == BeforeImageControl) ? beforeScaleTransform : afterScaleTransform;

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
            viewModel.UpdateZoomLevel(scaleTransform.ScaleX);

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
                isPanning = true;
                lastMousePosition = e.GetPosition(scrollViewer);
                scrollViewer.CaptureMouse();
            }
        }

        private void ImageScrollViewer_MouseMove(object sender, MouseEventArgs e)
        {
            ScrollViewer scrollViewer = (ScrollViewer)sender;
            if (isPanning)
            {
                Point currentMousePosition = e.GetPosition(scrollViewer);
                double deltaX = currentMousePosition.X - lastMousePosition.X;
                double deltaY = currentMousePosition.Y - lastMousePosition.Y;

                scrollViewer.ScrollToHorizontalOffset(scrollViewer.HorizontalOffset - deltaX);
                scrollViewer.ScrollToVerticalOffset(scrollViewer.VerticalOffset - deltaY);

                lastMousePosition = currentMousePosition;
            }
        }

        private void ImageScrollViewer_MouseUp(object sender, MouseButtonEventArgs e)
        {
            ScrollViewer scrollViewer = (ScrollViewer)sender;
            if (e.ChangedButton == MouseButton.Left)
            {
                isPanning = false;
                scrollViewer.ReleaseMouseCapture();
            }
        }

        

        private void RoiCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (DataContext is MainViewModel viewModel)
            {
                viewModel.RoiViewModel.MouseDownCommand.Execute(e);
            }
        }

        private void RoiCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (DataContext is MainViewModel viewModel)
            {
                viewModel.RoiViewModel.MouseMoveCommand.Execute(e);
            }
        }

        private void RoiCanvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (DataContext is MainViewModel viewModel)
            {
                viewModel.RoiViewModel.MouseUpCommand.Execute(e);
            }
        }

        // Helper method to find an ancestor of a specific type in the visual tree
        private static T FindAncestor<T>(DependencyObject current) where T : DependencyObject
        {
            do
            {
                current = VisualTreeHelper.GetParent(current);
                if (current is T ancestor) return ancestor;
            } while (current != null);
            return null;
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {

        }
    }
}