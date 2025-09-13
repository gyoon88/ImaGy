using System;
using System.Windows;
using ImaGy.ViewModels;

namespace ImaGy.Services
{
    public class ImageDisplayService : BaseViewModel
    {
        // --- Zoom and Scale ---
        private double currentZoomScale = 1.0;
        public double CurrentZoomScale
        {
            get => currentZoomScale;
            set => SetProperty(ref currentZoomScale, value);
        }

        private double initialZoomScale = 1.0;
        public double InitialZoomScale
        {
            get => initialZoomScale;
            private set => SetProperty(ref initialZoomScale, value);
        }

        // --- ScrollViewer State ---
        private double horizontalOffset;
        public double HorizontalOffset { get => horizontalOffset; set => SetProperty(ref horizontalOffset, value); }

        private double verticalOffset;
        public double VerticalOffset { get => verticalOffset; set => SetProperty(ref verticalOffset, value); }

        private double viewportWidth;
        public double ViewportWidth { get => viewportWidth; set => SetProperty(ref viewportWidth, value); }

        private double viewportHeight;
        public double ViewportHeight { get => viewportHeight; set => SetProperty(ref viewportHeight, value); }


        public Action<double, double>? RequestScrollAction { get; set; }

        private Point lastMousePositionForPan;
        private bool isPanning;

        public void Zoom(double delta, Point mousePosition)
        {
            double oldScale = CurrentZoomScale;

            // 1. Calculate new scale
            double zoomFactor = 1.1;
            double newScale = oldScale;
            if (delta > 0) newScale *= zoomFactor;
            else newScale /= zoomFactor;

            if (newScale < 0.05) newScale = 0.05;
            if (newScale > 50.0) newScale = 50.0;

            // 2. Get the point on the full-resolution image under the mouse
            Point pointToKeepCentered = new Point(
                (HorizontalOffset + mousePosition.X) / oldScale,
                (VerticalOffset + mousePosition.Y) / oldScale
            );

            // 3. Apply the new scale
            CurrentZoomScale = newScale;

            // 4. Calculate new offsets to keep the point centered
            double newHorizontalOffset = (pointToKeepCentered.X * newScale) - mousePosition.X;
            double newVerticalOffset = (pointToKeepCentered.Y * newScale) - mousePosition.Y;

            // 5. Request the scroll viewer to scroll to the new offsets
            RequestScrollAction?.Invoke(newHorizontalOffset, newVerticalOffset);
        }

        public void PanMouseDown(Point mousePosition)
        {
            lastMousePositionForPan = mousePosition;
            isPanning = true;
        }

        public void PanMouseMove(Point currentMousePosition)
        {
            if (isPanning)
            {
                double deltaX = currentMousePosition.X - lastMousePositionForPan.X;
                double deltaY = currentMousePosition.Y - lastMousePositionForPan.Y;

                RequestScrollAction?.Invoke(HorizontalOffset - deltaX, VerticalOffset - deltaY);
            }
        }

        public void PanMouseUp()
        {
            isPanning = false;
        }

        public void ResetDisplay(double imagePixelWidth, double imagePixelHeight, double viewerWidth, double viewerHeight)
        {
            if (imagePixelWidth == 0 || viewerWidth == 0)
            {
                CurrentZoomScale = 1.0;
                InitialZoomScale = 1.0;
                return;
            }

            double scaleX = viewerWidth / imagePixelWidth;
            double scaleY = viewerHeight / imagePixelHeight;
            double fitToScreenScale = Math.Min(scaleX, scaleY);

            CurrentZoomScale = fitToScreenScale;
            InitialZoomScale = fitToScreenScale;

            RequestScrollAction?.Invoke(0, 0); // Scroll to home
        }
    }
}