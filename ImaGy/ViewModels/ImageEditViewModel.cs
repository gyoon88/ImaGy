using System;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using ImaGy.Services;
using ImaGy.ViewModels.Commands;

namespace ImaGy.ViewModels
{
    public sealed class ImageEditViewModel : BaseViewModel
    {
        private readonly MainViewModel _main;
        private readonly LoggingService _logging;
        private readonly byte[] _baselinePixels;
        private readonly int _width;
        private readonly int _height;
        private readonly int _stride;
        private readonly double _dpiX;
        private readonly double _dpiY;
        private readonly bool _wasGray8;

        private readonly DispatcherTimer _previewDebounce;

        private double brightness;
        public double Brightness
        {
            get => brightness;
            set
            {
                if (SetProperty(ref brightness, value))
                    SchedulePreview();
            }
        }

        private double contrast = 1.0;
        public double Contrast
        {
            get => contrast;
            set
            {
                if (SetProperty(ref contrast, value))
                    SchedulePreview();
            }
        }

        private double gamma = 1.0;
        public double Gamma
        {
            get => gamma;
            set
            {
                if (SetProperty(ref gamma, value))
                    SchedulePreview();
            }
        }

        private BitmapSource? previewBitmap;
        public BitmapSource? PreviewBitmap
        {
            get => previewBitmap;
            private set
            {
                if (!SetProperty(ref previewBitmap, value))
                    return;
                CommandManager.InvalidateRequerySuggested();
            }
        }

        private bool isPreviewBusy;
        public bool IsPreviewBusy
        {
            get => isPreviewBusy;
            private set => SetProperty(ref isPreviewBusy, value);
        }

        public ICommand ResetCommand { get; }
        public ICommand ApplyCommand { get; }
        public ICommand CancelCommand { get; }

        private Action? _closeWindow;

        public ImageEditViewModel(MainViewModel mainViewModel, LoggingService loggingService)
        {
            _main = mainViewModel;
            _logging = loggingService;

            var src = mainViewModel.AfterImage ?? throw new InvalidOperationException("이미지가 없습니다.");
            var cap = ImageToneAdjustService.CapturePbgra32(src);
            _baselinePixels = cap.Pixels;
            _width = cap.Width;
            _height = cap.Height;
            _stride = cap.Stride;
            _dpiX = cap.DpiX;
            _dpiY = cap.DpiY;
            _wasGray8 = cap.WasGray8;

            _previewDebounce = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(85) };
            _previewDebounce.Tick += (_, _) =>
            {
                _previewDebounce.Stop();
                _ = RunPreviewAsync();
            };

            ResetCommand = new RelayCommand(ResetToDefaults);
            ApplyCommand = new RelayCommand(Apply, () => !IsPreviewBusy && PreviewBitmap != null);
            CancelCommand = new RelayCommand(() => _closeWindow?.Invoke());

            _ = RunPreviewAsync();
        }

        public void RegisterCloseHandler(Action closeWindow) => _closeWindow = closeWindow;

        private void ResetToDefaults()
        {
            Brightness = 0;
            Contrast = 1;
            Gamma = 1;
            SchedulePreview();
        }

        private void SchedulePreview()
        {
            _previewDebounce.Stop();
            _previewDebounce.Start();
        }

        private async Task RunPreviewAsync()
        {
            IsPreviewBusy = true;
            CommandManager.InvalidateRequerySuggested();
            try
            {
                double b = Brightness;
                double c = Contrast;
                double g = Gamma;
                byte[] baseline = _baselinePixels;
                int stride = _stride;
                int height = _height;
                int width = _width;
                double dpiX = _dpiX;
                double dpiY = _dpiY;

                byte[] outPixels = await Task.Run(() =>
                    ImageToneAdjustService.TransformPbgra(baseline, stride, height, b, c, g)).ConfigureAwait(true);

                var bmp = ImageToneAdjustService.CreatePbgraBitmap(outPixels, width, height, stride, dpiX, dpiY);
                PreviewBitmap = bmp;
            }
            finally
            {
                IsPreviewBusy = false;
                CommandManager.InvalidateRequerySuggested();
            }
        }

        private void Apply()
        {
            if (PreviewBitmap == null)
                return;

            var display = ImageToneAdjustService.ToDisplayFormat(PreviewBitmap, _wasGray8);
            _main.undoRedoService.AddState(_main.AfterImage);
            _main.AfterImage = display;
            _main.historyService.AddHistory("Image edit (밝기·대비·감마)", 0);
            _logging.AddLog("Image edit 적용: 밝기·대비·감마.");
            _closeWindow?.Invoke();
        }
    }
}
