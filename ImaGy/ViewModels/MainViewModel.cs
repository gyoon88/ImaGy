using ImaGy.Models;
using ImaGy.Services;
using ImaGy.View;
using ImaGy.ViewModels.Commands;
using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.ViewModels
{
    public class MainViewModel : BaseViewModel
    {
        // --- 서비스 및 모델 ---
        public ImageDisplayService ImageDisplay { get; }
        private readonly FilterProcessor filterProcessor;
        private readonly MatchingProcessor matchingProcessor;
        private readonly ColorContrastProcess colorContrastProcess;
        private readonly MorphologyProcessor morphologyProcessor;
        public readonly UndoRedoService<BitmapSource?> undoRedoService;
        public readonly HistoryService historyService;
        private readonly LoggingService loggingService;
        private readonly ClipboardImageService clipboardService;
        private readonly FileService fileService;
        private readonly ImageProcessingService imageProcessingService;
        private readonly HistogramService histogramService;
        
       

        private readonly CropService cropService;


        private bool isProcessing;
        private BitmapSource? beforeImage;
        private BitmapSource? afterImage;
        private BitmapSource? templateImage;
        public ProcessingParameters Parameters { get; }


        private bool isColor;
        public bool IsColor
        {
            get => isColor;
            private set => SetProperty(ref isColor, value); // 외부에서는 값을 변경할 수 없도록 private set
        }
        // --- 속성 ---
        public BitmapSource? BeforeImage
        {
            get => beforeImage;
            set => SetProperty(ref beforeImage, value);
        }
        public BitmapSource? AfterImage
        {
            get => afterImage;
            set        
            {
                if (SetProperty(ref afterImage, value)) // 값이 변경되었을 때만 실행
                {
                    //CurrentImage가 바뀔 때 IsColor 속성도 함께 업데이트
                    IsColor = afterImage?.Format == PixelFormats.Bgra32;
                }
            }
        }
        public BitmapSource? TemplateImage
        {
            get => templateImage;
            set => SetProperty(ref templateImage, value);
        }
        private string? fileName;
        public string? FileName
        {
            get => fileName;
            set => SetProperty(ref fileName, value);
        }
        private string? imageResolution;
        public string? ImageResolution
        {
            get => imageResolution;
            set => SetProperty(ref imageResolution, value);
        }
        private string? mouseCoordinates;
        public string? MouseCoordinates
        {
            get => mouseCoordinates;
            set => SetProperty(ref mouseCoordinates, value);
        }
        private string? processingTime;
        public string? ProcessingTime
        {
            get => processingTime;
            set => SetProperty(ref processingTime, value);
        }
        
        public bool IsProcessing
        {
            get => isProcessing;
            set => SetProperty(ref isProcessing, value);
        }

        public bool IsImageLoading { get; set; }           


        public string LogText => loggingService.LogText;
        public ObservableCollection<string> HistoryItems => historyService.HistoryItems;
        

        public string ZoomLevel
        {
            get
            {
                if (ImageDisplay.InitialZoomScale == 0) return "100%";
                double relativeZoom = (ImageDisplay.CurrentZoomScale / ImageDisplay.InitialZoomScale) * 100;
                return $"{relativeZoom:F0}%";
            }
        }

        // --- 커맨드 ---
        public ICommand ZoomCommand { get; }
        public ICommand PanMouseDownCommand { get; }
        public ICommand PanMouseMoveCommand { get; }
        public ICommand PanMouseUpCommand { get; }
        public ICommand UpdateMouseCoordinatesCommand { get; }
        public ICommand ClearMouseCoordinatesCommand { get; }
        public ICommand OpenImageCommand { get; }
        public ICommand SaveImageCommand { get; }
        public ICommand UndoCommand { get; }
        public ICommand RedoCommand { get; }
        public ICommand FilterringCommand { get; }
        public ICommand ImageMatchingCommand { get; }
        public ICommand ViewHistogramCommand { get; }
        public ICommand ExportHistoryCommand { get; }
        public ICommand ExportLogCommand { get; }
        public ICommand OpenTemplateImageCommand { get; }
        public ICommand CopyImageCommand { get; }
        public ICommand PasteImageCommand { get; }
        
        public ICommand MinimapCommand { get; }
        public ICommand ApplyCropCommand { get; }
        public ICommand OpenGridWorkbenchCommand { get; }

        public ImageViewerInteractionService InteractionService { get; }

        

        public MainViewModel()
        {
            Parameters = new ProcessingParameters();
            // --- 서비스 초기화 ---
            ImageDisplay = new ImageDisplayService();
            undoRedoService = new UndoRedoService<BitmapSource?>();
            loggingService = new LoggingService();
            filterProcessor = new FilterProcessor();
            morphologyProcessor = new MorphologyProcessor();
            matchingProcessor = new MatchingProcessor();
            colorContrastProcess = new ColorContrastProcess();
            historyService = new HistoryService();
            clipboardService = new ClipboardImageService();
            fileService = new FileService();
            imageProcessingService = new ImageProcessingService(
                colorContrastProcess, matchingProcessor, filterProcessor, morphologyProcessor,
                undoRedoService, historyService, loggingService
                );
            histogramService = new HistogramService();
            cropService = new CropService();
            InteractionService = new ImageViewerInteractionService(this, ImageDisplay, cropService, historyService);

            // --- 커맨드 초기화 ---
            // Mouse Command
            ZoomCommand = new RelayCommand<MouseWheelEventArgs>(e => ImageDisplay.Zoom(e.Delta, e.GetPosition((IInputElement)e.Source)));
            PanMouseDownCommand = new RelayCommand<MouseButtonEventArgs>(InteractionService.MouseDown);
            PanMouseMoveCommand = new RelayCommand<System.Windows.Input.MouseEventArgs>(InteractionService.MouseMove);
            PanMouseUpCommand = new RelayCommand<MouseButtonEventArgs>(InteractionService.MouseUp);
            UpdateMouseCoordinatesCommand = new RelayCommand<System.Windows.Input.MouseEventArgs>(e =>
            {
                if (e?.OriginalSource is System.Windows.Controls.Image img)
                    UpdateMousePixelReadout(img, e.GetPosition(img));
            });
            ClearMouseCoordinatesCommand = new RelayCommand(() => ClearMouseCoordinates());
            // File Command
            OpenImageCommand = new OpenImageCommand(this, fileService, loggingService);
            SaveImageCommand = new SaveImageCommand(this, fileService, loggingService);
            OpenTemplateImageCommand = new OpenTemplateImageCommand(this, fileService, loggingService);

            UndoCommand = new UndoCommand(this, undoRedoService);
            RedoCommand = new RedoCommand(this, undoRedoService);
            ExportHistoryCommand = new ExportHistoryCommand(this, historyService, loggingService, fileService);
            ExportLogCommand = new ExportLogCommand(this, loggingService, fileService);
            CopyImageCommand = new CopyImageCommand(this, clipboardService);
            PasteImageCommand = new PasteImageCommand(this, clipboardService);

            // Processing Command
            FilterringCommand = new ApplyFilterCommand(this, imageProcessingService);
            ImageMatchingCommand = new ApplyImageMatchingCommand(this, imageProcessingService);
            ViewHistogramCommand = new ViewHistogramCommand(this, histogramService);
            ApplyCropCommand = new ApplyCropCommand(this);
            OpenGridWorkbenchCommand = new RelayCommand(() =>
            {
                var vm = new GridWorkbenchViewModel(loggingService);
                var win = new GridWorkbenchWindow { DataContext = vm };
                win.Show();
            });
            MinimapCommand = new RelayCommand(() => {
                MinimapViewModel minimapViewModel = new MinimapViewModel(this);
                ImaGy.View.MinimapWindow minimapWindow = new ImaGy.View.MinimapWindow
                {
                    DataContext = minimapViewModel
                };
                minimapWindow.Show();
            }, () => AfterImage != null);

            // --- 속성 변경 이벤트 구독 ---
            ImageDisplay.PropertyChanged += (s, e) => {
                if (e.PropertyName == nameof(ImageDisplay.CurrentZoomScale) || e.PropertyName == nameof(ImageDisplay.InitialZoomScale))
                {
                    OnPropertyChanged(nameof(ZoomLevel)); // ZoomLevel UI 업데이트
                }
            };
            
            loggingService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(LoggingService.LogText))
                {
                    OnPropertyChanged(nameof(LogText));
                }
            };

            // 파일 속성 업데이트
            {
                FileName = "No file loaded";
                ImageResolution = "N/A";
                ProcessingTime = "0 ms";
                MouseCoordinates = "X: -, Y: -";
            }

        }

        // 메인뷰어 이미지 조작 메서드
        /// <summary>
        /// 이미지 위 마우스: 레이아웃(확대) 좌표를 픽셀 인덱스로 환산한 뒤 GV(밝기)·RGB를 상태줄에 표시합니다.
        /// </summary>
        public void UpdateMousePixelReadout(System.Windows.Controls.Image image, System.Windows.Point positionInImage)
        {
            if (image.Source is not BitmapSource bmp || bmp.PixelWidth <= 0 || bmp.PixelHeight <= 0)
            {
                MouseCoordinates = "X: -, Y: -";
                return;
            }

            double s = ImageDisplay.CurrentZoomScale;
            if (s < 1e-12) s = 1.0;

            int px = (int)Math.Floor(positionInImage.X / s);
            int py = (int)Math.Floor(positionInImage.Y / s);
            if (px < 0 || py < 0 || px >= bmp.PixelWidth || py >= bmp.PixelHeight)
            {
                MouseCoordinates = "이미지 밖";
                return;
            }

            if (!TrySamplePixel(bmp, px, py, out byte r, out byte g, out byte b, out _))
            {
                MouseCoordinates = $"X: {px}, Y: {py} | GV: —";
                return;
            }

            double gray = 0.299 * r + 0.587 * g + 0.114 * b;
            bool singleChannel = bmp.Format == PixelFormats.Gray8
                                 || bmp.Format == PixelFormats.Gray16
                                 || bmp.Format == PixelFormats.Gray32Float
                                 || bmp.Format == PixelFormats.Indexed8;

            if (singleChannel)
                MouseCoordinates = string.Format(CultureInfo.InvariantCulture, "X: {0}, Y: {1} | GV: {2}", px, py, r);
            else
                MouseCoordinates = string.Format(CultureInfo.InvariantCulture,
                    "X: {0}, Y: {1} | GV: {2:F0} (R{3} G{4} B{5})", px, py, gray, r, g, b);
        }

        private static bool TrySamplePixel(BitmapSource bmp, int px, int py, out byte r, out byte g, out byte b, out byte a)
        {
            r = g = b = 0;
            a = 255;
            try
            {
                var rect = new Int32Rect(px, py, 1, 1);
                if (bmp.Format == PixelFormats.Gray8)
                {
                    var buf = new byte[1];
                    bmp.CopyPixels(rect, buf, 1, 0);
                    r = g = b = buf[0];
                    return true;
                }

                var cropped = new CroppedBitmap(bmp, rect);
                var conv = new FormatConvertedBitmap(cropped, PixelFormats.Pbgra32, null, 0);
                var p = new byte[4];
                conv.CopyPixels(p, 4, 0);
                b = p[0];
                g = p[1];
                r = p[2];
                a = p[3];
                return true;
            }
            catch
            {
                return false;
            }
        }

        public void ClearMouseCoordinates()
        {
            MouseCoordinates = "X: -, Y: -";
        }

        public void ShowTemplateImageViewer()

        {

            TemplateImageViewerViewModel templateViewerViewModel = new TemplateImageViewerViewModel(this);

            TemplateImageViewer templateViewer = new TemplateImageViewer

            {

                DataContext = templateViewerViewModel

            };

            templateViewer.Show();

        }

    }
}