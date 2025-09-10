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
        private bool isSyncing = false; // Flag to prevent event recursion
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
            if (imageSource == null || BeforeImageScrollViewer.ActualWidth == 0) return;

            // Calculate the scale required to fit the image to the viewport
            double scaleX = BeforeImageScrollViewer.ActualWidth / imageSource.PixelWidth;
            double scaleY = BeforeImageScrollViewer.ActualHeight / imageSource.PixelHeight;
            double fitScale = Math.Min(scaleX, scaleY);

            // The "base" scale for 100% display is always the fit-to-screen scale.
            viewModel.BaseScale = fitScale;

            // The actual initial scale should not be more than 1.0 (1:1 pixel ratio).
            double initialAbsoluteScale = Math.Min(1.0, fitScale);

            // Apply the initial scale to both image views
            beforeScaleTransform.ScaleX = initialAbsoluteScale;
            beforeScaleTransform.ScaleY = initialAbsoluteScale;
            afterScaleTransform.ScaleX = initialAbsoluteScale;
            afterScaleTransform.ScaleY = initialAbsoluteScale;

            // Reset scroll positions
            BeforeImageScrollViewer.ScrollToHome();
            AfterImageScrollViewer.ScrollToHome();

            // Update the ZoomLevel display based on the new base scale
            viewModel.UpdateZoomLevel(initialAbsoluteScale);
        }

        private void BeforeImageScrollViewer_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            if (isSyncing) return;
            isSyncing = true;

            AfterImageScrollViewer.ScrollToHorizontalOffset(e.HorizontalOffset);
            AfterImageScrollViewer.ScrollToVerticalOffset(e.VerticalOffset);

            if (DataContext is MainViewModel viewModel)
            {
                viewModel.ScrollViewerHorizontalOffset = e.HorizontalOffset;
                viewModel.ScrollViewerVerticalOffset = e.VerticalOffset;
                viewModel.ScrollViewerViewportWidth = e.ViewportWidth;
                viewModel.ScrollViewerViewportHeight = e.ViewportHeight;
            }
            isSyncing = false;
        }

        private void AfterImageScrollViewer_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            if (isSyncing) return;
            isSyncing = true;

            BeforeImageScrollViewer.ScrollToHorizontalOffset(e.HorizontalOffset);
            BeforeImageScrollViewer.ScrollToVerticalOffset(e.VerticalOffset);

            if (DataContext is MainViewModel viewModel)
            {
                viewModel.ScrollViewerHorizontalOffset = e.HorizontalOffset;
                viewModel.ScrollViewerVerticalOffset = e.VerticalOffset;
                viewModel.ScrollViewerViewportWidth = e.ViewportWidth;
                viewModel.ScrollViewerViewportHeight = e.ViewportHeight;
            }
            isSyncing = false;
        }

        private void ImageControl_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            e.Handled = true; 

            double zoomFactor = 1.1;
            double newScale;

            if (e.Delta > 0)
            {
                newScale = beforeScaleTransform.ScaleX * zoomFactor;
            }
            else
            {
                newScale = beforeScaleTransform.ScaleX / zoomFactor;
            }

            // Apply the new scale to both transforms to keep them in sync
            beforeScaleTransform.ScaleX = newScale;
            beforeScaleTransform.ScaleY = newScale;
            afterScaleTransform.ScaleX = newScale;
            afterScaleTransform.ScaleY = newScale;

            viewModel.UpdateZoomLevel(newScale);

            // Also update the coordinates in the status bar
            Image_MouseMove(sender, e);
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

        private void Image_MouseMove(object sender, MouseEventArgs e)
        {
            if (viewModel == null || sender is not Image imageControl || imageControl.Source == null)
            {
                return;
            }

            ScaleTransform scaleTransform = (imageControl == BeforeImageControl) ? beforeScaleTransform : afterScaleTransform;
            Point mousePos = e.GetPosition(imageControl);

            // Translate coordinates from control space to image pixel space
            int actualX = (int)(mousePos.X / scaleTransform.ScaleX);
            int actualY = (int)(mousePos.Y / scaleTransform.ScaleY);

            // Ensure coordinates are within the image bounds
            if (actualX >= 0 && actualX < imageControl.Source.Width && actualY >= 0 && actualY < imageControl.Source.Height)
            {
                viewModel.UpdateMouseCoordinates(actualX, actualY);
            }
            else
            {
                viewModel.ClearMouseCoordinates();
            }
        }

        private void Image_MouseLeave(object sender, MouseEventArgs e)
        {
            if (viewModel != null)
            {
                viewModel.ClearMouseCoordinates();
            }
        }
    }
}