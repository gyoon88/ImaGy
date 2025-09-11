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

        // Properties to hold the actual size of the minimap control from the View
        private double _minimapActualWidth;
        public double MinimapActualWidth
        {
            get => _minimapActualWidth;
            set 
            {
                SetProperty(ref _minimapActualWidth, value);
                UpdateViewport();
            }
        }

        private double _minimapActualHeight;
        public double MinimapActualHeight
        {
            get => _minimapActualHeight;
            set
            {
                SetProperty(ref _minimapActualHeight, value);
                UpdateViewport();
            }
        }

        // Scaled properties for the viewport rectangle to be used by the View
        private double _scaledViewportX;
        public double ScaledViewportX
        {
            get => _scaledViewportX;
            set => SetProperty(ref _scaledViewportX, value);
        }

        private double _scaledViewportY;
        public double ScaledViewportY
        {
            get => _scaledViewportY;
            set => SetProperty(ref _scaledViewportY, value);
        }

        private double _scaledViewportWidth;
        public double ScaledViewportWidth
        {
            get => _scaledViewportWidth;
            set => SetProperty(ref _scaledViewportWidth, value);
        }

        private double _scaledViewportHeight;
        public double ScaledViewportHeight
        {
            get => _scaledViewportHeight;
            set => SetProperty(ref _scaledViewportHeight, value);
        }

        public MinimapViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _mainViewModel.PropertyChanged += MainViewModel_PropertyChanged;
            MouseDownCommand = new RelayCommand<object>(ExecuteMouseDown);
            MouseMoveCommand = new RelayCommand<object>(ExecuteMouseMove);
            MouseUpCommand = new RelayCommand<object>(ExecuteMouseUp);
        }

        private void MainViewModel_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.AfterImage) || e.PropertyName == nameof(MainViewModel.BeforeImage))
            {
                OnPropertyChanged(nameof(MainImage));
                UpdateViewport();
            }
            else if (
                e.PropertyName == nameof(MainViewModel.ScrollViewerHorizontalOffset) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerVerticalOffset) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerViewportWidth) ||
                e.PropertyName == nameof(MainViewModel.ScrollViewerViewportHeight) ||
                e.PropertyName == nameof(MainViewModel.ZoomLevel))
            {
                UpdateViewport();
            }
        }

        private void UpdateViewport()
        {
            if (MainImage == null || MinimapActualWidth == 0 || MinimapActualHeight == 0)
            {
                ScaledViewportWidth = 0;
                ScaledViewportHeight = 0;
                return;
            }

            // 1. Calculate the scale of the image as it's displayed in the minimap
            double ratioX = MinimapActualWidth / MainImage.PixelWidth;
            double ratioY = MinimapActualHeight / MainImage.PixelHeight;
            double minimapScale = Math.Min(ratioX, ratioY);

            // 2. Calculate the size of the displayed image in the minimap (due to Stretch="Uniform")
            double displayedImageWidth = MainImage.PixelWidth * minimapScale;
            double displayedImageHeight = MainImage.PixelHeight * minimapScale;

            // 3. Calculate the letterbox/pillarbox offsets
            double offsetX = (MinimapActualWidth - displayedImageWidth) / 2;
            double offsetY = (MinimapActualHeight - displayedImageHeight) / 2;

            // 4. Get the main viewport properties (in original image coordinates)
            double mainZoom = _mainViewModel.ImageDisplay.CurrentZoomScale;
            if (mainZoom <= 0) return;

            double viewportX = _mainViewModel.ScrollViewerHorizontalOffset / mainZoom;
            double viewportY = _mainViewModel.ScrollViewerVerticalOffset / mainZoom;
            double viewportWidth = _mainViewModel.ScrollViewerViewportWidth / mainZoom;
            double viewportHeight = _mainViewModel.ScrollViewerViewportHeight / mainZoom;

            // 5. Scale the viewport to the minimap's displayed image size and apply offset
            ScaledViewportX = viewportX * minimapScale + offsetX;
            ScaledViewportY = viewportY * minimapScale + offsetY;
            ScaledViewportWidth = viewportWidth * minimapScale;
            ScaledViewportHeight = viewportHeight * minimapScale;
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
                // This needs to be done via a callback or event to MainWindow.xaml.cs
                // _mainViewModel.ScrollMainImage(scrollDeltaX, scrollDeltaY); 

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
