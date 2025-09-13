using ImaGy.Services;
using System.ComponentModel;
using System.Windows.Media.Imaging;

namespace ImaGy.ViewModels
{
    public class MinimapViewModel : BaseViewModel
    {
        private readonly MainViewModel mainViewModel;
        private readonly ImageDisplayService imageDisplayService;

        public BitmapSource? MainImage => mainViewModel.AfterImage;

        private double minimapActualWidth;
        public double MinimapActualWidth
        {
            get => minimapActualWidth;
            set
            {
                if (SetProperty(ref minimapActualWidth, value))
                {
                    UpdateViewport();
                }
            }
        }

        private double minimapActualHeight;
        public double MinimapActualHeight
        {
            get => minimapActualHeight;
            set
            {
                if (SetProperty(ref minimapActualHeight, value))
                {
                    UpdateViewport();
                }
            }
        }

        private double scaledViewportX;
        public double ScaledViewportX { get => scaledViewportX; set => SetProperty(ref scaledViewportX, value); }

        private double scaledViewportY;
        public double ScaledViewportY { get => scaledViewportY; set => SetProperty(ref scaledViewportY, value); }

        private double scaledViewportWidth;
        public double ScaledViewportWidth { get => scaledViewportWidth; set => SetProperty(ref scaledViewportWidth, value); }

        private double scaledViewportHeight;
        public double ScaledViewportHeight { get => scaledViewportHeight; set => SetProperty(ref scaledViewportHeight, value); }

        public MinimapViewModel(MainViewModel mainViewModel)
        {
            this.mainViewModel = mainViewModel;
            this.imageDisplayService = mainViewModel.ImageDisplay;

            this.imageDisplayService.PropertyChanged += ImageDisplayService_PropertyChanged;

            UpdateViewport();
        }

        private void ImageDisplayService_PropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            // Listen for any property change that affects the viewport
            UpdateViewport();
        }

        private void UpdateViewport()
        {
            if (MainImage == null || MinimapActualWidth == 0 || MinimapActualHeight == 0) return;

            double currentZoom = imageDisplayService.CurrentZoomScale;
            double initialZoom = imageDisplayService.InitialZoomScale;

            if (currentZoom <= initialZoom)
            {
                ScaledViewportWidth = 0;
                ScaledViewportHeight = 0;
                return;
            }

            double scale = System.Math.Min(MinimapActualWidth / MainImage.PixelWidth, MinimapActualHeight / MainImage.PixelHeight);

            double displayedImageWidth = MainImage.PixelWidth * scale;
            double displayedImageHeight = MainImage.PixelHeight * scale;
            double offsetX = (MinimapActualWidth - displayedImageWidth) / 2;
            double offsetY = (MinimapActualHeight - displayedImageHeight) / 2;

            double originalViewportWidth = imageDisplayService.ViewportWidth / currentZoom;
            double originalViewportHeight = imageDisplayService.ViewportHeight / currentZoom;
            double originalViewportX = imageDisplayService.HorizontalOffset / currentZoom;
            double originalViewportY = imageDisplayService.VerticalOffset / currentZoom;

            double desiredX = originalViewportX * scale + offsetX;
            double desiredY = originalViewportY * scale + offsetY;
            double desiredWidth = originalViewportWidth * scale;
            double desiredHeight = originalViewportHeight * scale;

            double validX = offsetX;
            double validY = offsetY;
            double validWidth = displayedImageWidth;
            double validHeight = displayedImageHeight;

            double finalX = System.Math.Max(desiredX, validX);
            double finalY = System.Math.Max(desiredY, validY);

            double finalRight = System.Math.Min(desiredX + desiredWidth, validX + validWidth);
            double finalBottom = System.Math.Min(desiredY + desiredHeight, validY + validHeight);

            double finalWidth = finalRight - finalX;
            double finalHeight = finalBottom - finalY;

            ScaledViewportX = finalX;
            ScaledViewportY = finalY;
            ScaledViewportWidth = (finalWidth < 0) ? 0 : finalWidth;
            ScaledViewportHeight = (finalHeight < 0) ? 0 : finalHeight;
        }

        public void Cleanup()
        {
            imageDisplayService.PropertyChanged -= ImageDisplayService_PropertyChanged;
        }
    }
}
