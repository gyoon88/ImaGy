using System;
using System.Windows;
using ImaGy.ViewModels;

namespace ImaGy.Services
{
    public class ImageDisplayService : BaseViewModel
    {
        private double _currentZoomScale = 1.0;
        public double CurrentZoomScale
        {
            get => _currentZoomScale;
            set => SetProperty(ref _currentZoomScale, value);
        }

        public Action<double, double>? RequestScrollAction { get; set; }

        private Point _lastMousePositionForPan;
        private bool _isPanning;

        public void Zoom(double delta, Point mousePosition)
        {
            double zoomFactor = 1.1;
            double newScale = CurrentZoomScale;

            if (delta > 0) newScale *= zoomFactor;
            else newScale /= zoomFactor;

            if (newScale < 0.05) newScale = 0.05;
            if (newScale > 50.0) newScale = 50.0;
            
            CurrentZoomScale = newScale;
        }

        public void PanMouseDown(Point mousePosition)
        {
            _lastMousePositionForPan = mousePosition;
            _isPanning = true;
        }

        public void PanMouseMove(Point currentMousePosition, ScrollViewerInfo scrollInfo)
        {
            if (_isPanning)
            {
                double deltaX = currentMousePosition.X - _lastMousePositionForPan.X;
                double deltaY = currentMousePosition.Y - _lastMousePositionForPan.Y;

                RequestScrollAction?.Invoke(scrollInfo.HorizontalOffset - deltaX, scrollInfo.VerticalOffset - deltaY);
            }
        }

        public void PanMouseUp()
        {
            _isPanning = false;
        }

        public void ResetDisplay(double imagePixelWidth, double imagePixelHeight, double viewerActualWidth, double viewerActualHeight)
        {
            if (imagePixelWidth == 0 || viewerActualWidth == 0)
            {
                CurrentZoomScale = 1.0;
                return;
            }

            double scaleX = viewerActualWidth / imagePixelWidth;
            double scaleY = viewerActualHeight / imagePixelHeight;
            CurrentZoomScale = Math.Min(scaleX, scaleY);

            RequestScrollAction?.Invoke(0, 0); // 스크롤 홈으로
        }
    }

    // ScrollViewer 정보를 전달하기 위한 간단한 구조체
    public struct ScrollViewerInfo
    {
        public double HorizontalOffset { get; set; }
        public double VerticalOffset { get; set; }
    }
}