using ImaGy.Models;
using ImaGy.Services;
using ImaGy.View;
using ImaGy.ViewModels.Commands;
using System;
using System.Collections.ObjectModel;
using System.Windows;
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
        private readonly RoiViewModel roiViewModel;
        private readonly CropService cropService;


        private bool isProcessing;
        private BitmapSource? beforeImage;
        private BitmapSource? afterImage;
        private BitmapSource? templateImage;
        private int threshold = 128;
        private int kernelSize = 3;
        private double sigma = 1.0;
        private bool useCircularKernel = false;


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
        public int Threshold
        {
            get => threshold;
            set => SetProperty(ref threshold, value);
        }
        public int KernelSize
        {
            get => kernelSize;
            set => SetProperty(ref kernelSize, value);
        }
        public double Sigma
        {
            get => sigma;
            set => SetProperty(ref sigma, value);
        }
        public bool UseCircularKernel
        {
            get => useCircularKernel;
            set => SetProperty(ref useCircularKernel, value);
        }
        public bool IsProcessing
        {
            get => isProcessing;
            set => SetProperty(ref isProcessing, value);
        }

        public bool IsImageLoading { get; set; }

        private bool isInCropMode;
        public bool IsInCropMode
        {
            get => isInCropMode;
            set => SetProperty(ref isInCropMode, value);
        }

        private Rect cropRectangle;
        public Rect CropRectangle
        {
            get => cropRectangle;
            set => SetProperty(ref cropRectangle, value);
        }

        private Point cropStartPoint;


        public string LogText => loggingService.LogText;
        public ObservableCollection<string> HistoryItems => historyService.HistoryItems;
        public RoiViewModel RoiViewModel => roiViewModel;

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
        public ICommand SelectRoiCommand { get; }
        public ICommand MinimapCommand { get; }
        public ICommand ApplyCropCommand { get; }

        

        public MainViewModel()
        {
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

            // --- 커맨드 초기화 ---
            // Mouse Command
            ZoomCommand = new RelayCommand<MouseWheelEventArgs>(e => {
                if (e.OriginalSource is FrameworkElement element)
                {
                    ImageDisplay.Zoom(e.Delta, e.GetPosition(element));
                }
            });

            PanMouseDownCommand = new RelayCommand<MouseButtonEventArgs>(e =>
            {
                if (e.OriginalSource is FrameworkElement element)
                {
                    if (IsInCropMode)
                    {
                        element.CaptureMouse();
                        cropStartPoint = e.GetPosition(element);
                        CropRectangle = new Rect(cropStartPoint, cropStartPoint);
                    }
                    else if (element.CaptureMouse())
                    {
                        ImageDisplay.PanMouseDown(e.GetPosition(element));
                    }
                }
            });

            PanMouseMoveCommand = new RelayCommand<MouseEventArgs>(e =>
            {
                if (e.OriginalSource is FrameworkElement element)
                {
                    if (IsInCropMode && e.LeftButton == MouseButtonState.Pressed)
                    {
                        Point currentPoint = e.GetPosition(element);
                        CropRectangle = new Rect(cropStartPoint, currentPoint);
                    }
                    else
                    {
                        ImageDisplay.PanMouseMove(e.GetPosition(element));
                    }
                    UpdateMouseCoordinates((int)e.GetPosition(element).X, (int)e.GetPosition(element).Y);
                }
            });

            PanMouseUpCommand = new RelayCommand<MouseButtonEventArgs>(e =>
            {
                if (e.OriginalSource is FrameworkElement element)
                {
                    if (IsInCropMode)
                    {
                        element.ReleaseMouseCapture();
                        if (CropRectangle.Width > 0 && CropRectangle.Height > 0 && BeforeImage != null)
                        {
                            // Convert screen coordinates to image coordinates
                            double scale = ImageDisplay.CurrentZoomScale;
                            var roi = new RoiModel(
                                CropRectangle.X / scale,
                                CropRectangle.Y / scale,
                                CropRectangle.Width / scale,
                                CropRectangle.Height / scale
                            );

                            var cropped = cropService.CropImage(BeforeImage, roi);
                            if (cropped != null)
                            {
                                BeforeImage = cropped;
                                AfterImage = cropped;
                                historyService.AddHistory("Crop", 0);
                            }
                        }
                        IsInCropMode = false;
                        CropRectangle = new Rect(); // Reset rectangle
                    }
                    else
                    {
                        element.ReleaseMouseCapture();
                        ImageDisplay.PanMouseUp();
                    }
                }
            });

            UpdateMouseCoordinatesCommand = new RelayCommand<MouseEventArgs>(e => {
                if (e.OriginalSource is FrameworkElement element && BeforeImage != null)
                {
                    Point pos = e.GetPosition(element);
                    int x = (int)(pos.X);
                    int y = (int)(pos.Y);
                    UpdateMouseCoordinates(x, y);
                }
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
        public void UpdateMouseCoordinates(int x, int y)
        {
            if (BeforeImage == null) return;
            MouseCoordinates = $"X: {x}, Y: {y}";
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