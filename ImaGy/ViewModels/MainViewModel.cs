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
        private readonly ImageProcessor imageProcessor;
        private readonly ImageProcessorSSE imageProcessorSSE;
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



        // --- 속성 ---
        public BitmapSource? BeforeImage
        {
            get => beforeImage;
            set
            {
                if (SetProperty(ref beforeImage, value) && value != null)
                {
                    ImageLoadedCallback?.Invoke(value);
                }
            }
        }
        public BitmapSource? AfterImage
        {
            get => afterImage;
            set => SetProperty(ref afterImage, value);
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


        public string LogText => loggingService.LogText;
        public ObservableCollection<string> HistoryItems => historyService.HistoryItems;
        public RoiViewModel RoiViewModel => roiViewModel;

        // ZoomLevel은 이제 ImageDisplayService의 값을 기반으로 계산
        public string ZoomLevel => $"{ImageDisplay.CurrentZoomScale * 100:F0}%";

        // ScrollViewer properties for Minimap & Pan
        private double scrollViewerViewportWidth;
        public double ScrollViewerViewportWidth
        {
            get => scrollViewerViewportWidth;
            set => SetProperty(ref scrollViewerViewportWidth, value);
        }
        private double scrollViewerViewportHeight;
        public double ScrollViewerViewportHeight
        {
            get => scrollViewerViewportHeight;
            set => SetProperty(ref scrollViewerViewportHeight, value);
        }
        private double scrollViewerHorizontalOffset;
        public double ScrollViewerHorizontalOffset
        {
            get => scrollViewerHorizontalOffset;
            set => SetProperty(ref scrollViewerHorizontalOffset, value);
        }
        private double scrollViewerVerticalOffset;
        public double ScrollViewerVerticalOffset
        {
            get => scrollViewerVerticalOffset;
            set => SetProperty(ref scrollViewerVerticalOffset, value);
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
        public ICommand ColorContrastCommand { get; }
        public ICommand FilterringCommand { get; }
        public ICommand MorphorogyCommand { get; }
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

        public Action<BitmapSource>? ImageLoadedCallback { get; set; }

        public MainViewModel()
        {
            // --- 서비스 초기화 ---
            ImageDisplay = new ImageDisplayService();
            undoRedoService = new UndoRedoService<BitmapSource?>();
            loggingService = new LoggingService();
            imageProcessor = new ImageProcessor();
            imageProcessorSSE = new ImageProcessorSSE();
            filterProcessor = new FilterProcessor();
            morphologyProcessor = new MorphologyProcessor();
            matchingProcessor = new MatchingProcessor();
            colorContrastProcess = new ColorContrastProcess();
            historyService = new HistoryService();
            clipboardService = new ClipboardImageService();
            fileService = new FileService();
            imageProcessingService = new ImageProcessingService(
                imageProcessor, imageProcessorSSE,  
                colorContrastProcess, matchingProcessor, filterProcessor, morphologyProcessor,
                undoRedoService, historyService, loggingService
                );
            histogramService = new HistogramService();
            roiViewModel = new RoiViewModel();
            cropService = new CropService();

            // --- 커맨드 초기화 ---
            ZoomCommand = new RelayCommand<MouseWheelEventArgs>(e => {
                if (e.OriginalSource is FrameworkElement element)
                {
                    ImageDisplay.Zoom(e.Delta, e.GetPosition(element));
                }
            });

            PanMouseDownCommand = new RelayCommand<MouseButtonEventArgs>(e => {
                if (e.OriginalSource is FrameworkElement element && element.CaptureMouse())
                {
                    ImageDisplay.PanMouseDown(e.GetPosition(element));
                }
            });

            PanMouseMoveCommand = new RelayCommand<MouseEventArgs>(e => {
                if (e.OriginalSource is FrameworkElement element)
                {
                    var scrollInfo = new ScrollViewerInfo { HorizontalOffset = ScrollViewerHorizontalOffset, VerticalOffset = ScrollViewerVerticalOffset };
                    ImageDisplay.PanMouseMove(e.GetPosition(element), scrollInfo);
                }
            });

            PanMouseUpCommand = new RelayCommand<MouseButtonEventArgs>(e => {
                 if (e.OriginalSource is FrameworkElement element)
                {
                    element.ReleaseMouseCapture();
                    ImageDisplay.PanMouseUp();
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

            OpenImageCommand = new OpenImageCommand(this, fileService, loggingService);
            SaveImageCommand = new SaveImageCommand(this, fileService, loggingService);
            UndoCommand = new UndoCommand(this, undoRedoService);
            RedoCommand = new RedoCommand(this, undoRedoService);


            ColorContrastCommand = new ApplyFilterCommand(this, imageProcessingService);
            FilterringCommand = new ApplyFilterCommand(this, imageProcessingService);
            MorphorogyCommand = new ApplyFilterCommand(this, imageProcessingService);
            ImageMatchingCommand = new ApplyImageMatchingCommand(this, imageProcessingService);
            ViewHistogramCommand = new ViewHistogramCommand(this, histogramService);


            ExportHistoryCommand = new ExportHistoryCommand(this, historyService, loggingService, fileService);
            ExportLogCommand = new ExportLogCommand(this, loggingService, fileService);
            OpenTemplateImageCommand = new OpenTemplateImageCommand(this, fileService, loggingService);
            CopyImageCommand = new CopyImageCommand(this, clipboardService);
            PasteImageCommand = new PasteImageCommand(this, clipboardService);

            SelectRoiCommand = new RelayCommand(() => { RoiViewModel.IsDrawingRoi = !RoiViewModel.IsDrawingRoi; });

            MinimapCommand = new RelayCommand(() => {
                MinimapViewModel minimapViewModel = new MinimapViewModel(this);
                ImaGy.View.MinimapWindow minimapWindow = new ImaGy.View.MinimapWindow
                {
                    DataContext = minimapViewModel
                };
                minimapWindow.Show();
            }, () => AfterImage != null);

            ApplyCropCommand = new RelayCommand(() => {
                if (AfterImage != null && RoiViewModel.CurrentRoi != null)
                {
                    var croppedImage = cropService.CropImage(AfterImage, RoiViewModel.CurrentRoi);
                    if (croppedImage != null)
                    {
                        AfterImage = croppedImage;
                        BeforeImage = croppedImage;
                        RoiViewModel.CurrentRoi = null;
                        RoiViewModel.IsDrawingRoi = false;
                    }
                }
            }, () => RoiViewModel.CurrentRoi != null && AfterImage != null);

            // --- 속성 변경 이벤트 구독 ---
            ImageDisplay.PropertyChanged += (s, e) => {
                if (e.PropertyName == nameof(ImageDisplay.CurrentZoomScale))
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

            // Initialize Properties
            FileName = "No file loaded";
            ImageResolution = "N/A";
            ProcessingTime = "0 ms";
            MouseCoordinates = "X: -, Y: -";
        }

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
        // Other methods like ShowTemplateImageViewer, ExecuteSelectRoi etc. can be kept
    }
}