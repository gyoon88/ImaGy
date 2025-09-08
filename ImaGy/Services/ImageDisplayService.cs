using System;
using System.Windows;
using System.Windows.Media.Imaging;
using ImaGy.ViewModels; // For BaseViewModel if ImageDisplayService inherits from it

namespace ImaGy.Services
{
    public class ImageDisplayService : BaseViewModel // Inherit from BaseViewModel for PropertyChanged
    {
        private double _currentZoomScale;
        public double CurrentZoomScale
        {
            get => _currentZoomScale;
            set => SetProperty(ref _currentZoomScale, value);
        }

        private double _imageHorizontalOffset;
        public double ImageHorizontalOffset
        {
            get => _imageHorizontalOffset;
            set => SetProperty(ref _imageHorizontalOffset, value);
        }

        private double _imageVerticalOffset;
        public double ImageVerticalOffset
        {
            get => _imageVerticalOffset;
            set => SetProperty(ref _imageVerticalOffset, value);
        }

        // Callback for View to request scroll (e.g., ScrollToHome, ScrollToOffset)
        // Parameters: newHorizontalOffset, newVerticalOffset, viewportWidth, viewportHeight
        public Action<double, double, double, double>? RequestScrollAction { get; set; }

        private Point _lastMousePositionForPan;
        private bool _isPanning;

        public ImageDisplayService()
        {
            CurrentZoomScale = 1.0; // Initial zoom
            ImageHorizontalOffset = 0;
            ImageVerticalOffset = 0;
        }

        public void Zoom(double delta, double imageActualWidth, double imageActualHeight, Point mousePosition)
        {
            double zoomFactor = 1.1;
            double newScale = CurrentZoomScale;

            if (delta > 0) // Zoom in
            {
                newScale *= zoomFactor;
            }
            else // Zoom out
            {
                newScale /= zoomFactor;
            }

            // Prevent extreme zooming
            if (newScale < 0.1) newScale = 0.1;
            if (newScale > 10.0) newScale = 10.0;

            // Calculate new offsets to keep mouse position centered
            double mouseXRelativeToImage = (ImageHorizontalOffset + mousePosition.X) / CurrentZoomScale;
            double mouseYRelativeToImage = (ImageVerticalOffset + mousePosition.Y) / CurrentZoomScale;

            ImageHorizontalOffset = (mouseXRelativeToImage * newScale) - mousePosition.X;
            ImageVerticalOffset = (mouseYRelativeToImage * newScale) - mousePosition.Y;

            CurrentZoomScale = newScale;

            // Request the view to scroll to the new calculated offsets
            RequestScrollAction?.Invoke(ImageHorizontalOffset, ImageVerticalOffset, 0, 0); // Viewport dimensions not needed here
        }

        public void PanMouseDown(Point mousePosition)
        {
            _lastMousePositionForPan = mousePosition;
            _isPanning = true;
        }

        public void PanMouseMove(Point currentMousePosition)
        {
            if (_isPanning)
            {
                double deltaX = currentMousePosition.X - _lastMousePositionForPan.X;
                double deltaY = currentMousePosition.Y - _lastMousePositionForPan.Y;

                ImageHorizontalOffset -= deltaX;
                ImageVerticalOffset -= deltaY;

                _lastMousePositionForPan = currentMousePosition;

                // Request the view to scroll to the new calculated offsets
                RequestScrollAction?.Invoke(ImageHorizontalOffset, ImageVerticalOffset, 0, 0);
            }
        }

        public void PanMouseUp()
        {
            _isPanning = false;
        }

        public void ResetDisplay(double imagePixelWidth, double imagePixelHeight, double viewerActualWidth, double viewerActualHeight)
        {
            if (imagePixelWidth == 0 || imagePixelHeight == 0 || viewerActualWidth == 0 || viewerActualHeight == 0)
            {
                CurrentZoomScale = 1.0;
                ImageHorizontalOffset = 0;
                ImageVerticalOffset = 0;
                RequestScrollAction?.Invoke(0, 0, 0, 0); // Request scroll to home
                return;
            }

            double scaleX = viewerActualWidth / imagePixelWidth;
            double scaleY = viewerActualHeight / imagePixelHeight;
            double initialScale = Math.Min(scaleX, scaleY);

            CurrentZoomScale = initialScale;
            ImageHorizontalOffset = 0;
            ImageVerticalOffset = 0;

            RequestScrollAction?.Invoke(0, 0, 0, 0); // Request scroll to home
        }

        // Method for minimap to request pan
        public void PanFromMinimap(double deltaX, double deltaY)
        {
            ImageHorizontalOffset += deltaX;
            ImageVerticalOffset += deltaY;
            RequestScrollAction?.Invoke(ImageHorizontalOffset, ImageVerticalOffset, 0, 0);
        }
    }
}