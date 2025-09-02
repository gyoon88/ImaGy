using ImaGy.Model;
using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using ImaGy.View;

namespace ImaGy.ViewModel
{
    internal class MainViewModel : BaseViewModel
    {
        // Fields
        private BitmapSource? beforeImage;
        private BitmapSource? afterImage;
        private string? fileName;
        private string? imageResolution;
        private string? zoomLevel;
        private string? processingTime;

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
        public string LogText => loggingService.LogText;

        // Model Class Instances
        private readonly ImageProcessor imageProcessor;
        private readonly UndoRedoService<BitmapSource?> undoRedoService;
        private readonly HistoryService historyService;
        private readonly LoggingService loggingService;
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

        public Action<BitmapSource>? ImageLoadedCallback { get; set; }

        // Constructor
        public MainViewModel()
        {
            // Service Initialization
            undoRedoService = new UndoRedoService<BitmapSource?>();
            loggingService = new LoggingService();
            imageProcessor = new ImageProcessor();
            historyService = new HistoryService();

            // Commands
            OpenImageCommand = new RelayCommand(ExecuteOpenImage);
            SaveImageCommand = new RelayCommand(ExecuteSaveImage, _ => AfterImage != null);
            UndoCommand = new RelayCommand(ExecuteUndo, _ => undoRedoService.CanUndo);
            RedoCommand = new RelayCommand(ExecuteRedo, _ => undoRedoService.CanRedo);

            ColorContrastCommand = new RelayCommand(ExecuteColorContrast, _ => BeforeImage != null);
            FilterringCommand = new RelayCommand(ExecuteFilterring, _ => BeforeImage != null);
            MorphorogyCommand = new RelayCommand(ExecuteMorphorogy, _ => BeforeImage != null);
            ImageMatchingCommand = new RelayCommand(ExecuteImageMatching, _ => BeforeImage != null);
            ViewHistogramCommand = new RelayCommand(ExecuteViewHistogram, _ => AfterImage != null);

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
        }


        // Image Open-Save
        private void ExecuteOpenImage(object? parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpeg;*.jpg;*.bmp)|*.png;*.jpeg;*.jpg;*.bmp|All files (*.*)|*.*"
            };
            if (openDialog.ShowDialog() == true)
            {
                try
                {
                    double elapsedMs = ProcessTime.Measure(() =>
                    {
                        var bitmap = new BitmapImage();
                        bitmap.BeginInit();
                        bitmap.UriSource = new Uri(openDialog.FileName);
                        bitmap.CacheOption = BitmapCacheOption.OnLoad;
                        bitmap.EndInit();
                        bitmap.Freeze();

                        BeforeImage = bitmap;
                        AfterImage = bitmap; // Set AfterImage to the new image
                        FileName = Path.GetFileName(openDialog.FileName);
                        ImageResolution = $"{bitmap.PixelWidth}x{bitmap.PixelHeight}";

                        // Notify the View to reset zoom
                        ImageLoadedCallback?.Invoke(bitmap);

                        // Clear history for new image
                        undoRedoService.Clear();
                        historyService.Clear();
                        loggingService.AddLog($"Image loaded: {FileName}");
                        CommandManager.InvalidateRequerySuggested();
                    });
                    ProcessingTime = $"Load Time: {elapsedMs:F2} ms";
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to load the image.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void ExecuteSaveImage(object? parameter)
        {
            if (AfterImage == null) return;

            SaveFileDialog saveDialog = new SaveFileDialog
            {
                Filter = "PNG Image|*.png|JPEG Image|*.jpg|BMP Image|*.bmp"
            };

            if (saveDialog.ShowDialog() == true)
            {
                try
                {
                    BitmapEncoder? encoder = Path.GetExtension(saveDialog.FileName).ToLower() switch
                    {
                        ".png" => new PngBitmapEncoder(),
                        ".jpg" or ".jpeg" => new JpegBitmapEncoder(),
                        ".bmp" => new BmpBitmapEncoder(),
                        _ => null
                    };

                    if (encoder != null)
                    {
                        encoder.Frames.Add(BitmapFrame.Create(AfterImage));
                        using var fileStream = new FileStream(saveDialog.FileName, FileMode.Create);
                        encoder.Save(fileStream);
                        loggingService.AddLog($"Image saved: {Path.GetFileName(saveDialog.FileName)}");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to save the image.\n\nError: {ex.Message}",
                     "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // Applies an image processing function to the current image
        private void ApplyProcessing(string historyMessage, Func<BitmapSource, BitmapSource> processFunction)
        {
            var imageToProcess = AfterImage ?? BeforeImage;
            if (imageToProcess == null) return;

            try
            {
                // 1. Add current state to undo stack
                undoRedoService.AddState(imageToProcess);

                // 2. Process image and measure time
                var (processedImage, elapsedMs) = ProcessTime.Measure(() => processFunction(imageToProcess));

                // 3. Update UI
                AfterImage = processedImage;
                ProcessingTime = $"{historyMessage}: {elapsedMs:F2} ms";

                // 4. Add to history and log
                historyService.AddHistory(historyMessage, elapsedMs);
                loggingService.AddLog($"{historyMessage} applied.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred during image processing.\n\nError: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // Command Implementations
        private void ExecuteColorContrast(object? parameter)
        {
            string? processCommand = parameter?.ToString();
            switch (processCommand)
            {
                case "Bin":
                    ApplyProcessing("Binarization", image => imageProcessor.ApplyBinarization(image, 128));
                    break;
                case "Bin_SSE":
                    ApplyProcessing("Binarization", image => imageProcessor.ApplyBinarization(image, 128));
                    break;

                case "Equal":
                    ApplyProcessing("Equalization", image => imageProcessor.ApplyEqualization(image));
                    break;
            }
        }

        private void ExecuteFilterring(object? parameter)
        {
            string? processCommand = parameter?.ToString();
            switch (processCommand)
            {
                case "Diff":
                    ApplyProcessing("Differential", image => imageProcessor.ApplyDifferential(image));
                    break;
                case "Sobel":
                    ApplyProcessing("Sobel", image => imageProcessor.ApplySobel(image));
                    break;
                case "Laplace":
                    ApplyProcessing("Laplacian", image => imageProcessor.ApplyLaplacian(image));
                    break;
                case "Average":
                    ApplyProcessing("Average Blur", image => imageProcessor.ApplyAverageBlur(image));
                    break;
                case "Gaussian":
                    ApplyProcessing("Gaussian", image => imageProcessor.ApplyGaussianBlur(image));
                    break;
            }
        }
        private void ExecuteMorphorogy(object? parameter)
        {
            string? processCommand = parameter?.ToString();
            switch (processCommand)
            {
                case "Dilation":
                    ApplyProcessing("Dilation", image => imageProcessor.ApplyDilation(image));
                    break;

                case "Erosion":
                    ApplyProcessing("Erosion", image => imageProcessor.ApplyErosion(image));
                    break;
            }
        }
        private void ExecuteImageMatching(object? parameter)
        {
            string? processCommand = parameter?.ToString();
            switch (processCommand)
            {
                case "NCC":
                    ApplyProcessing("NCC", image => imageProcessor.ApplyNCC(image));
                    break;

                case "SAD":
                    ApplyProcessing("SAD", image => imageProcessor.ApplySAD(image));
                    break;
                case "SSD":
                    ApplyProcessing("SSD", image => imageProcessor.ApplySSD(image));
                    break;
            }
        }

        private void ExecuteViewHistogram(object? parameter)
        {
            if (AfterImage == null) return;

            int[] histogramData = ServeHistogram.CalculateGrayscaleHistogram(AfterImage);
            HistogramViewModel histogramViewModel = new HistogramViewModel(histogramData);

            HistogramWindow histogramWindow = new HistogramWindow
            {
                DataContext = histogramViewModel
            };
            histogramWindow.Show();
        }

        // Undo/Redo
        private void ExecuteUndo(object? parameter)
        {
            AfterImage = undoRedoService.Undo(AfterImage);
        }

        private void ExecuteRedo(object? parameter)
        {
            AfterImage = undoRedoService.Redo(AfterImage);
        }

        public void UpdateZoomLevel(double scale)
        {
            ZoomLevel = $"{scale * 100:F0}%";
        }
    }
}