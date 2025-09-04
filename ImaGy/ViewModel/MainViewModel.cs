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
        public string LogText => loggingService.LogText;

        // Model Class Instances
        private readonly ImageProcessor imageProcessor;
        private readonly ImageProcessorSSE imageProcessorSSE;
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
        public ICommand ExportHistoryCommand { get; }
        public ICommand ExportLogCommand { get; }
        public ICommand OpenTemplateImageCommand { get; }

        public Action<BitmapSource>? ImageLoadedCallback { get; set; }

        // Constructor
        public MainViewModel()
        {
            // Service Initialization
            undoRedoService = new UndoRedoService<BitmapSource?>();
            loggingService = new LoggingService();
            imageProcessor = new ImageProcessor();
            imageProcessorSSE = new ImageProcessorSSE();
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
            ExportHistoryCommand = new RelayCommand(ExecuteExportHistory, _ => HistoryItems.Any());
            ExportLogCommand = new RelayCommand(ExecuteExportLog, _ => !string.IsNullOrEmpty(LogText));
            OpenTemplateImageCommand = new RelayCommand(ExecuteOpenTemplateImage);

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
                        bitmap.EndInit(); // 파일을 프로그램으로 업로드하여 다른 프로그램에서 접근 하지 못하는 현상을 제거
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

        private void ExecuteOpenTemplateImage(object? parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpeg;*.jpg;*.bmp)|*.png;*.jpeg;*.jpg;*.bmp|All files (*.*)|*.*"
            };
            if (openDialog.ShowDialog() == true)
            {
                try
                {
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(openDialog.FileName);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();
                    bitmap.Freeze();

                    TemplateImage = bitmap;
                    loggingService.AddLog($"Template image loaded: {Path.GetFileName(openDialog.FileName)}");

                    TemplateImageViewerViewModel templateViewerViewModel = new TemplateImageViewerViewModel(this);
                    TemplateImageViewer templateViewer = new TemplateImageViewer
                    {
                        DataContext = templateViewerViewModel
                    };
                    templateViewer.Show();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to load the template image.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // Applies an image processing function to the current image
        private async void ApplyProcessing(string processName, Func<BitmapSource, BitmapSource> processAction)
        {
            var imageToProcess = AfterImage ?? BeforeImage;
            if (imageToProcess == null)
            {
                loggingService.AddLog("No image loaded for processing.");
                MessageBox.Show("Please load an image first.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            try
            {
                undoRedoService.AddState(imageToProcess);
                BitmapSource? processedImage = null;
                double elapsedMs = 0;

                (processedImage, elapsedMs) = await Task.Run(() =>
                {
                    return ProcessTime.Measure(() => processAction(imageToProcess));
                });

                AfterImage = processedImage;
                ProcessingTime = $"{processName}: {elapsedMs:F2} ms";
                CommandManager.InvalidateRequerySuggested();

                historyService.AddHistory(processName, elapsedMs);
                loggingService.AddLog($"{processName} applied.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred during image processing.\n\nError: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                loggingService.AddLog($"Error applying {processName}: {ex.Message}");
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
                case "Diff_SSE":
                    ApplyProcessing("Differential_SSE", image => imageProcessorSSE.ApplyDifferentialSse(image));
                    break;

                case "Sobel":
                    ApplyProcessing("Sobel", image => imageProcessor.ApplySobel(image));
                    break;
                case "Sobel_SSE":
                    ApplyProcessing("Sobel_SSE", image => imageProcessorSSE.ApplySobelSse(image));
                    break;

                case "Laplace":
                    ApplyProcessing("Laplacian", image => imageProcessor.ApplyLaplacian(image));
                    break;
                case "Laplace_SSE":
                    ApplyProcessing("Laplacian_SSE", image => imageProcessorSSE.ApplyLaplacianSse(image));
                    break;

                case "Average":
                    ApplyProcessing("Average Blur", image => imageProcessor.ApplyAverageBlur(image));
                    break;
                case "Average_SSE":
                    ApplyProcessing("Average Blur_SSE", image => imageProcessorSSE.ApplyAverageBlurSse(image));
                    break;

                case "Gaussian":
                    ApplyProcessing("Gaussian", image => imageProcessor.ApplyGaussianBlur(image));
                    break;
                case "Gaussian_SSE":
                    ApplyProcessing("Gaussian_SSE", image => imageProcessorSSE.ApplyGaussianBlurSse(image));
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
                case "Dilation_SSE":
                    ApplyProcessing("Dilation_SSE", image => imageProcessorSSE.ApplyDilationSse(image));
                    break;

                case "Erosion":
                    ApplyProcessing("Erosion", image => imageProcessor.ApplyErosion(image));
                    break;
                case "Erosion_SSE":
                    ApplyProcessing("Erosion_SSE", image => imageProcessorSSE.ApplyErosionSse(image));
                    break;
            }
        }
        private async void ExecuteImageMatching(object? parameter) // Changed to async void
        {
            string? processCommand = parameter?.ToString();
            var imageToProcess = AfterImage ?? BeforeImage;

            if (imageToProcess == null)
            {
                loggingService.AddLog("No image loaded for matching.");
                MessageBox.Show("Please load an image first.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            if (TemplateImage == null)
            {
                loggingService.AddLog("No template image loaded for matching.");
                MessageBox.Show("Please load a template image first.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // Optional: Disable UI elements to prevent re-entry or indicate busy state
            // ImageMatchingCommand.CanExecute = false; // This requires a CanExecute parameter in RelayCommand

            try
            {
                // 1. Add current state to undo stack (can be done on UI thread)
                undoRedoService.AddState(imageToProcess);

                BitmapSource? processedImage = null;
                double elapsedMs = 0;

                // Move the long-running operation to a background thread
                (processedImage, elapsedMs) = await Task.Run(() => // Task.Run을 await
                {
                    // 아래 코드는 백그라운드 스레드에서 실행
                    BitmapSource? tempProcessedImage = null;
                    double tempElapsedMs = 0;

                    switch (processCommand)
                    {
                        case "NCC":
                            (tempProcessedImage, tempElapsedMs) = ProcessTime.Measure(() =>
                                imageProcessor.ApplyNCC(imageToProcess, TemplateImage));
                            break;
                        case "SAD":
                            (tempProcessedImage, tempElapsedMs) = ProcessTime.Measure(() =>
                                imageProcessor.ApplySAD(imageToProcess, TemplateImage));
                            break;
                        case "SSD":
                            (tempProcessedImage, tempElapsedMs) = ProcessTime.Measure(() =>
                                imageProcessor.ApplySSD(imageToProcess, TemplateImage));
                            break;
                        default:
                            throw new InvalidOperationException($"Unknown image matching command: {processCommand}");
                    }
                    return (tempProcessedImage, tempElapsedMs); // 백그라운드 스레드에서 결과 반환
                });

                // await 후, 실행은 원래 컨텍스트(UI 스레드)에서 재개
                // 3. UI 업데이트
                AfterImage = processedImage;
                ProcessingTime = $"{processCommand}: {elapsedMs:F2} ms";
                CommandManager.InvalidateRequerySuggested();

                // 4. 기록 및 로그에 추가 (UI 스레드 또는 백그라운드에서 수행 가능하지만, UI 업데이트는 UI 스레드가 필요합니다)
                historyService.AddHistory(processCommand, elapsedMs); // processCommand를 기록 메시지로 사용
                loggingService.AddLog($"{processCommand} applied.");
            }
            catch (InvalidOperationException ex) // Task.Run에서 발생하는 특정 예외를 catch
            {
                MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                loggingService.AddLog($"Error: {ex.Message}");
            }
            catch (Exception ex) // 일반 예외 처리
            {
                MessageBox.Show($"An error occurred during image matching.\n\nError: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                loggingService.AddLog($"Error applying image matching ({processCommand}): {ex.Message}");
            }
            finally
            {
                // 선택 사항: UI 요소 다시 활성화
                // ImageMatchingCommand.CanExecute = true;
            }
        }

        private void ExecuteViewHistogram(object? parameter)
        {
            if (AfterImage == null) return;

            HistogramViewModel histogramViewModel = new HistogramViewModel(this);

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

        private void ExecuteExportHistory(object? parameter)
        {
            SaveFileDialog saveDialog = new SaveFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv",
                FileName = "History.csv"
            };

            if (saveDialog.ShowDialog() == true)
            {
                try
                {
                    historyService.ExportToCsv(saveDialog.FileName);
                    loggingService.AddLog($"History exported to: {saveDialog.FileName}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to export history.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    loggingService.AddLog($"Error exporting history: {ex.Message}");
                }
            }
        }

        private void ExecuteExportLog(object? parameter)
        {
            SaveFileDialog saveDialog = new SaveFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv",
                FileName = "Log.csv"
            };

            if (saveDialog.ShowDialog() == true)
            {
                try
                {
                    loggingService.ExportToCsv(saveDialog.FileName);
                    loggingService.AddLog($"Log exported to: {saveDialog.FileName}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to export log.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    loggingService.AddLog($"Error exporting log: {ex.Message}");
                }
            }
        }
    }
}