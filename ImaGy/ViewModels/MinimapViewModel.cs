using System.Windows.Media.Imaging;
using System.Windows.Input;
using System.Windows;
using System;
using System.Windows.Media; // Added for ScaleTransform
using ImaGy.ViewModels.Commands;

namespace ImaGy.ViewModels
{
    public class MinimapViewModel : BaseViewModel
    {
        private readonly MainViewModel _mainViewModel;

        public BitmapSource? MainImage => _mainViewModel.AfterImage ?? _mainViewModel.BeforeImage;

        // Properties for the viewport rectangle
        private double _viewportX;
        public double ViewportX
        {
            get => _viewportX;
            set => SetProperty(ref _viewportX, value);
        }

        private double _viewportY;
        public double ViewportY
        {
            get => _viewportY;
            set => SetProperty(ref _viewportY, value);
        }

        private double _viewportWidth;
        public double ViewportWidth
        {
            get => _viewportWidth;
            set => SetProperty(ref _viewportWidth, value);
        }

        private double _viewportHeight;
        public double ViewportHeight
        {
            get => _viewportHeight;
            set => SetProperty(ref _viewportHeight, value);
        }

        public MinimapViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _mainViewModel.PropertyChanged += MainViewModel_PropertyChanged;
            UpdateViewport();
            MouseDownCommand = new RelayCommand<object>(ExecuteMouseDown);
            MouseMoveCommand = new RelayCommand<object>(ExecuteMouseMove);
            MouseUpCommand = new RelayCommand<object>(ExecuteMouseUp);
        }

        private void MainViewModel_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.AfterImage) || e.PropertyName == nameof(MainViewModel.BeforeImage) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerHorizontalOffset) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerVerticalOffset) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerViewportWidth) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerViewportHeight) ||
                e.PropertyName == nameof(MainViewModel.ZoomLevel)) 
            {
                OnPropertyChanged(nameof(MainImage));
                UpdateViewport();
            }
        }

        private void UpdateViewport()
        {
            if (MainImage != null)
            {
                double scale = _mainViewModel.ImageDisplay.CurrentZoomScale;

                if (scale > 0) // Avoid division by zero
                {
                    ViewportX = _mainViewModel.ScrollViewerHorizontalOffset / scale;
                    ViewportY = _mainViewModel.ScrollViewerVerticalOffset / scale;
                    ViewportWidth = _mainViewModel.ScrollViewerViewportWidth / scale;
                    ViewportHeight = _mainViewModel.ScrollViewerViewportHeight / scale;
                }
            }
        }

        // Mouse handling for dragging the viewport
        private Point _lastMousePosition;
        private bool _isDragging;

        public ICommand MouseDownCommand { get; }
        public ICommand MouseMoveCommand { get; }
        public ICommand MouseUpCommand { get; }

        private void ExecuteMouseDown(object? parameter)
        {
            if (parameter is MouseButtonEventArgs e)
            {
                _lastMousePosition = e.GetPosition(e.Source as IInputElement);
                _isDragging = true;
                (e.Source as UIElement)?.CaptureMouse();
            }
        }

        private void ExecuteMouseMove(object? parameter)
        {
            if (_isDragging && parameter is MouseEventArgs e)
            {
                Point currentMousePosition = e.GetPosition(e.Source as IInputElement);
                double deltaX = currentMousePosition.X - _lastMousePosition.X;
                double deltaY = currentMousePosition.Y - _lastMousePosition.Y;

                // Convert delta from minimap coordinates to main image scrollviewer coordinates
                // This requires knowing the scale factor between minimap and main image
                // For now, a direct mapping (will need refinement)
                double scrollDeltaX = deltaX;
                double scrollDeltaY = deltaY;

                // Update main image scrollviewer
                // This needs to be done via a callback or event to MainWindow.xaml.cs
                // For now, I will call a method on MainViewModel, which will then call the view.
                // _mainViewModel.ScrollMainImage(scrollDeltaX, scrollDeltaY); // Removed as it caused compilation error after revert

                _lastMousePosition = currentMousePosition;
            }
        }

        private void ExecuteMouseUp(object? parameter)
        {
            if (_isDragging)
            {
                _isDragging = false;
                (parameter as UIElement)?.ReleaseMouseCapture();
            }
        }
    }
}