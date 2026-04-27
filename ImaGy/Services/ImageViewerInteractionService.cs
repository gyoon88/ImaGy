using ImaGy.Models;
using ImaGy.ViewModels;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace ImaGy.Services
{
    public class ImageViewerInteractionService : ViewModels.BaseViewModel
    {
        private readonly MainViewModel _mainViewModel;
        private readonly ImageDisplayService _imageDisplayService;
        private readonly CropService _cropService;
        private readonly HistoryService _historyService;

        private bool _isInCropMode;
        public bool IsInCropMode
        {
            get => _isInCropMode;
            set => SetProperty(ref _isInCropMode, value);
        }

        private Rect _cropRectangle;
        public Rect CropRectangle
        {
            get => _cropRectangle;
            set => SetProperty(ref _cropRectangle, value);
        }

        private System.Windows.Point _cropStartPoint;

        public ImageViewerInteractionService(MainViewModel mainViewModel, ImageDisplayService imageDisplayService, CropService cropService, HistoryService historyService)
        {
            _mainViewModel = mainViewModel;
            _imageDisplayService = imageDisplayService;
            _cropService = cropService;
            _historyService = historyService;
        }

        public void ToggleCropMode()
        {
            IsInCropMode = !IsInCropMode;
            if (IsInCropMode)
            {
                CropRectangle = new Rect();
            }
        }

        public void MouseDown(MouseButtonEventArgs e)
        {
            if (e.OriginalSource is FrameworkElement element)
            {
                if (IsInCropMode)
                {
                    element.CaptureMouse();
                    _cropStartPoint = e.GetPosition(element);
                    CropRectangle = new Rect(_cropStartPoint, _cropStartPoint);
                }
                else if (element.CaptureMouse())
                {
                    _imageDisplayService.PanMouseDown(e.GetPosition(element));
                }
            }
        }

        public void MouseMove(System.Windows.Input.MouseEventArgs e)
        {
            if (e.OriginalSource is FrameworkElement element)
            {
                if (IsInCropMode && e.LeftButton == MouseButtonState.Pressed)
                {
                    System.Windows.Point currentPoint = e.GetPosition(element);
                    CropRectangle = new Rect(_cropStartPoint, currentPoint);
                }
                else
                {
                    _imageDisplayService.PanMouseMove(e.GetPosition(element));
                }

                if (e.OriginalSource is System.Windows.Controls.Image img)
                    _mainViewModel.UpdateMousePixelReadout(img, e.GetPosition(img));
            }
        }

        public void MouseUp(MouseButtonEventArgs e)
        {
            if (e.OriginalSource is FrameworkElement element)
            {
                if (IsInCropMode)
                {
                    element.ReleaseMouseCapture();
                    if (CropRectangle.Width > 0 && CropRectangle.Height > 0 && _mainViewModel.BeforeImage != null)
                    {
                        var roi = new RoiModel(
                            CropRectangle.X,
                            CropRectangle.Y,
                            CropRectangle.Width,
                            CropRectangle.Height
                        );

                        var cropped = _cropService.CropImage(_mainViewModel.BeforeImage, roi);
                        if (cropped != null)
                        {
                            _mainViewModel.BeforeImage = cropped;
                            _mainViewModel.AfterImage = cropped;
                            _mainViewModel.ImageResolution = $"{cropped.PixelWidth}x{cropped.PixelHeight}";
                            _historyService.AddHistory("Crop", 0);
                        }
                    }
                    IsInCropMode = false;
                    CropRectangle = new Rect();
                }
                else
                {
                    element.ReleaseMouseCapture();
                    _imageDisplayService.PanMouseUp();
                }
            }
        }
    }
}
