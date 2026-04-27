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
            set
            {
                var prev = beforeImage;
                if (!SetProperty(ref beforeImage, value))
                    return;
                if (beforeImage == null || prev == null
                    || prev.PixelWidth != beforeImage.PixelWidth
                    || prev.PixelHeight != beforeImage.PixelHeight)
                    ClearAnalysisRoi();
            }
        }
        public BitmapSource? AfterImage
        {
            get => afterImage;
            set
            {
                var prev = afterImage;
                if (!SetProperty(ref afterImage, value))
                    return;
                IsColor = afterImage?.Format == PixelFormats.Bgra32;
                if (afterImage == null || prev == null
                    || prev.PixelWidth != afterImage.PixelWidth
                    || prev.PixelHeight != afterImage.PixelHeight)
                    ClearAnalysisRoi();
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
        public ICommand OpenImageEditCommand { get; }

        public ICommand MinimapCommand { get; }
        public ICommand ApplyCropCommand { get; }
        public ICommand OpenGridWorkbenchCommand { get; }
        public ICommand ClearAnalysisRoiCommand { get; }

        public ImageViewerInteractionService InteractionService { get; }

        private System.Windows.Int32Rect? _analysisRoiPixels;
        /// <summary>히스토그램·라인합·라인 프로파일에 쓰는 분석 영역(픽셀, After 또는 단일 Before 좌표계).</summary>
        public System.Windows.Int32Rect? AnalysisRoiPixels
        {
            get => _analysisRoiPixels;
            private set
            {
                if (!SetProperty(ref _analysisRoiPixels, value))
                    return;
                OnPropertyChanged(nameof(HasAnalysisRoi));
                OnPropertyChanged(nameof(AnalysisRoiHint));
                RefreshAnalysisRoiOverlayLayout();
            }
        }

        public bool HasAnalysisRoi => _analysisRoiPixels.HasValue;

        public string AnalysisRoiHint =>
            _analysisRoiPixels is { } r
                ? $"분석 ROI {r.Width}×{r.Height} @({r.X},{r.Y})"
                : "분석 ROI 없음 — 히스토그램은 전체 이미지";

        private double _dipPerPixelX = 1;
        private double _dipPerPixelY = 1;
        private double _roiOverlayLeft;
        private double _roiOverlayTop;
        private double _roiOverlayWidth;
        private double _roiOverlayHeight;

        public double RoiOverlayLeft { get => _roiOverlayLeft; private set => SetProperty(ref _roiOverlayLeft, value); }
        public double RoiOverlayTop { get => _roiOverlayTop; private set => SetProperty(ref _roiOverlayTop, value); }
        public double RoiOverlayWidth { get => _roiOverlayWidth; private set => SetProperty(ref _roiOverlayWidth, value); }
        public double RoiOverlayHeight { get => _roiOverlayHeight; private set => SetProperty(ref _roiOverlayHeight, value); }

        /// <summary>메인 뷰어 Image 레이아웃이 바뀐 뒤(줌·창 크기) 호출: DIP/픽셀 비율을 맞추고 분석 ROI 오버레이를 다시 그립니다.</summary>
        public void UpdateViewPixelScaleFromImage(System.Windows.Controls.Image? img)
        {
            if (img?.Source is not BitmapSource bmp || bmp.PixelWidth < 1 || bmp.PixelHeight < 1)
            {
                RefreshAnalysisRoiOverlayLayout();
                return;
            }

            if (img.ActualWidth < 1e-3 || img.ActualHeight < 1e-3)
                return;

            _dipPerPixelX = img.ActualWidth / bmp.PixelWidth;
            _dipPerPixelY = img.ActualHeight / bmp.PixelHeight;
            RefreshAnalysisRoiOverlayLayout();
        }

        public void RefreshAnalysisRoiOverlayLayout()
        {
            if (!_analysisRoiPixels.HasValue)
            {
                RoiOverlayLeft = 0;
                RoiOverlayTop = 0;
                RoiOverlayWidth = 0;
                RoiOverlayHeight = 0;
                return;
            }

            var r = _analysisRoiPixels.Value;
            RoiOverlayLeft = r.X * _dipPerPixelX;
            RoiOverlayTop = r.Y * _dipPerPixelY;
            RoiOverlayWidth = r.Width * _dipPerPixelX;
            RoiOverlayHeight = r.Height * _dipPerPixelY;
        }

        public void SetAnalysisRoiPixels(System.Windows.Int32Rect rectPixels)
        {
            var src = AfterImage ?? BeforeImage;
            if (src == null)
                return;
            AnalysisRoiPixels = MainImageRoiSampling.ClipToBitmap(rectPixels, src.PixelWidth, src.PixelHeight);
        }

        public void ClearAnalysisRoi() => AnalysisRoiPixels = null;

        

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
            OpenImageEditCommand = new OpenImageEditCommand(this, loggingService);

            // Processing Command
            FilterringCommand = new ApplyFilterCommand(this, imageProcessingService);
            ImageMatchingCommand = new ApplyImageMatchingCommand(this, imageProcessingService);
            ViewHistogramCommand = new ViewHistogramCommand(this, histogramService);
            ApplyCropCommand = new ApplyCropCommand(this);
            ClearAnalysisRoiCommand = new RelayCommand(
                () => ClearAnalysisRoi(),
                () => HasAnalysisRoi);
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
                    // ROI 오버레이는 Image.LayoutUpdated → UpdateViewPixelScaleFromImage 에서 갱신
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

            if (!MainImageRoiSampling.TryGetDipPerPixel(image, out var dppx, out var dppy) || dppx < 1e-12 || dppy < 1e-12)
            {
                MouseCoordinates = "X: -, Y: -";
                return;
            }

            int px = (int)Math.Floor(positionInImage.X / dppx);
            int py = (int)Math.Floor(positionInImage.Y / dppy);
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