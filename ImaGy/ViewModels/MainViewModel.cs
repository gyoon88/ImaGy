using ImaGy.Models;
using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using ImaGy.View;
using ImaGy.Services;
using ImaGy.ViewModels.Commands;
using System.Windows.Media;

namespace ImaGy.ViewModels
{
    public class MainViewModel : BaseViewModel
    {
        // Fields
        private BitmapSource? beforeImage;
        private BitmapSource? afterImage;
        private BitmapSource? templateImage;
        private string? fileName;
        private string? imageResolution;
        private string? zoomLevel;
        private string? processingTime;
        private bool isProcessing;
        private int threshold = 128;
        private double sigma = 1.0;
        private int kernelSize = 3;

        public bool IsProcessing
        {
            get => isProcessing;
            set
            {
                isProcessing = value;
                OnPropertyChanged(); // Assuming you have a base method for property change notifications
            }
        }
        // Properties
        public BitmapSource? BeforeImage
        {
            get => beforeImage;
            set => SetProperty(ref beforeImage, value);
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
        public string? FileName
        {
            get => fileName;
            set => SetProperty(ref fileName, value);
        }
        public string? ImageResolution
        {
            get => imageResolution;
            set => SetProperty(ref imageResolution, value);
        }
        public string? ZoomLevel
        {
            get => zoomLevel;
            set => SetProperty(ref zoomLevel, value);
        }
        public string? ProcessingTime
        {
            get => processingTime;
            set => SetProperty(ref processingTime, value);
        }
        public double Sigma
        {
            get => sigma;
            set => SetProperty(ref sigma, value);
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

        public string LogText => loggingService.LogText;

        // ScrollViewer properties for Minimap
        private double _scrollViewerViewportWidth;
        public double ScrollViewerViewportWidth
        {
            get => _scrollViewerViewportWidth;
            set => SetProperty(ref _scrollViewerViewportWidth, value);
        }

        private double _scrollViewerViewportHeight;
        public double ScrollViewerViewportHeight
        {
            get => _scrollViewerViewportHeight;
            set => SetProperty(ref _scrollViewerViewportHeight, value);
        }

        private double _scrollViewerHorizontalOffset;
        public double ScrollViewerHorizontalOffset
        {
            get => _scrollViewerHorizontalOffset;
            set => SetProperty(ref _scrollViewerHorizontalOffset, value);
        }

        private double _scrollViewerVerticalOffset;
        public double ScrollViewerVerticalOffset
        {
            get => _scrollViewerVerticalOffset;
            set => SetProperty(ref _scrollViewerVerticalOffset, value);
        }

        // Model Class Instances
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

        public ObservableCollection<string> HistoryItems => historyService.HistoryItems;

        // Commands
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

        public ScaleTransform beforeScaleTransform { get; set; }

        public RoiViewModel RoiViewModel => roiViewModel;

        // Constructor
        public MainViewModel()
        {
            // Service Initialization
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


            // Commands
            OpenImageCommand = new OpenImageCommand(this, fileService, loggingService);
            SaveImageCommand = new OpenImageCommand(this, fileService, loggingService);

            UndoCommand = new UndoCommand(this, undoRedoService);
            RedoCommand = new UndoCommand(this, undoRedoService);

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


            SelectRoiCommand = new RelayCommand(ExecuteSelectRoi);
            MinimapCommand = new RelayCommand(ExecuteMinimapCommand, _ => AfterImage != null);
            ApplyCropCommand = new RelayCommand(ExecuteApplyCrop, _ => RoiViewModel.CurrentRoi != null && AfterImage != null);

            // Event Subscription
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
            ZoomLevel = "100%";
            ProcessingTime = "0 ms";

            beforeScaleTransform = new ScaleTransform(); // Initialize
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

        public void UpdateZoomLevel(double scale)
        {
            ZoomLevel = $"{scale * 100:F0}%";
        }

        private void ExecuteSelectRoi(object? parameter)
        {
            RoiViewModel.IsDrawingRoi = !RoiViewModel.IsDrawingRoi;
        }

        private void ExecuteMinimapCommand(object? parameter)
        {
            MinimapViewModel minimapViewModel = new MinimapViewModel(this);
            MinimapWindow minimapWindow = new MinimapWindow
            {
                DataContext = minimapViewModel
            };
            minimapWindow.Show();
        }

        private void ExecuteApplyCrop(object? parameter)
        {
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
        }
    }
}