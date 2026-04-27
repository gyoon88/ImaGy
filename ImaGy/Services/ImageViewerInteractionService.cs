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

        private Rect _histogramRoiDraft;
        public Rect HistogramRoiDraft
        {
            get => _histogramRoiDraft;
            set => SetProperty(ref _histogramRoiDraft, value);
        }

        private bool _showHistogramRoiDraft;
        public bool ShowHistogramRoiDraft
        {
            get => _showHistogramRoiDraft;
            private set => SetProperty(ref _showHistogramRoiDraft, value);
        }

        private bool _histogramRoiDrag;
        private System.Windows.Point _histogramRoiStart;
        private System.Windows.Controls.Image? _histogramRoiImage;

        public ImageViewerInteractionService(MainViewModel mainViewModel, ImageDisplayService imageDisplayService, CropService cropService, HistoryService historyService)
        {
            _mainViewModel = mainViewModel;
            _imageDisplayService = imageDisplayService;
            _cropService = cropService;
            _historyService = historyService;
        }

        public void ToggleCropMode()
        {
            HistogramRoiDraft = new Rect();
            IsInCropMode = !IsInCropMode;
            if (IsInCropMode)
            {
                CropRectangle = new Rect();
            }
        }

        public void MouseDown(MouseButtonEventArgs e)
        {
            if (e.OriginalSource is not FrameworkElement element)
                return;

            bool shiftLeft = e.LeftButton == MouseButtonState.Pressed
                && (Keyboard.Modifiers & ModifierKeys.Shift) == ModifierKeys.Shift;
            if (!IsInCropMode && shiftLeft && e.OriginalSource is System.Windows.Controls.Image imgRoi)
            {
                BitmapSource? canonical = _mainViewModel.AfterImage ?? _mainViewModel.BeforeImage;
                if (canonical != null && ReferenceEquals(imgRoi.Source, canonical))
                {
                    imgRoi.CaptureMouse();
                    _histogramRoiImage = imgRoi;
                    _histogramRoiDrag = true;
                    _histogramRoiStart = e.GetPosition(imgRoi);
                    HistogramRoiDraft = new Rect(_histogramRoiStart, _histogramRoiStart);
                    ShowHistogramRoiDraft = true;
                }
                return;
            }

            if (IsInCropMode)
            {
                element.CaptureMouse();
                _cropStartPoint = e.GetPosition(element);
                CropRectangle = new Rect(_cropStartPoint, _cropStartPoint);
                return;
            }

            if (element.CaptureMouse())
                _imageDisplayService.PanMouseDown(e.GetPosition(element));
        }

        public void MouseMove(System.Windows.Input.MouseEventArgs e)
        {
            if (e.OriginalSource is not FrameworkElement element)
                return;

            if (_histogramRoiDrag && _histogramRoiImage is { } imgDrag)
            {
                HistogramRoiDraft = new Rect(_histogramRoiStart, e.GetPosition(imgDrag));
                return;
            }

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

        public void MouseUp(MouseButtonEventArgs e)
        {
            if (e.OriginalSource is not FrameworkElement element)
                return;

            if (_histogramRoiDrag && _histogramRoiImage is { } imgUp)
            {
                imgUp.ReleaseMouseCapture();
                _histogramRoiDrag = false;
                _histogramRoiImage = null;
                BitmapSource? canonical = _mainViewModel.AfterImage ?? _mainViewModel.BeforeImage;
                if (canonical != null
                    && ReferenceEquals(imgUp.Source, canonical)
                    && HistogramRoiDraft.Width >= 1
                    && HistogramRoiDraft.Height >= 1)
                {
                    var rectPixels = MainImageRoiSampling.ViewRectToPixelRect(
                        HistogramRoiDraft,
                        imgUp,
                        canonical.PixelWidth,
                        canonical.PixelHeight);
                    _mainViewModel.SetAnalysisRoiPixels(rectPixels);
                }

                HistogramRoiDraft = new Rect();
                ShowHistogramRoiDraft = false;
                return;
            }

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
