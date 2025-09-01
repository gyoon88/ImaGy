using ImaGy.Model;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;

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
        private ObservableCollection<string>? historyItems;
        private string? logText;

        private Stack<BitmapSource> undoStack;
        private Stack<BitmapSource> redoStack;

        // Properties
        public BitmapSource? BeforeImage
        {
            get => beforeImage;
            set { beforeImage = value; OnPropertyChanged(); }
        }
        public BitmapSource? AfterImage
        {
            get => afterImage;
            set { afterImage = value; OnPropertyChanged(); }
        }
        public string? FileName
        {
            get => fileName;
            set { fileName = value; OnPropertyChanged(); }
        }
        public string? ImageResolution
        {
            get => imageResolution;
            set { imageResolution = value; OnPropertyChanged(); }
        }
        public string? ZoomLevel
        {
            get => zoomLevel;
            set { zoomLevel = value; OnPropertyChanged(); }
        }
        public string? ProcessingTime
        {
            get => processingTime;
            set { processingTime = value; OnPropertyChanged(); }
        }
        public string LogText
        {
            get => logText;
            set { logText = value; OnPropertyChanged(); }
        }

        // Model Class Instances
        //private readonly ImageDocument imageDocument;
        //private readonly HistoryManager historyManager;
        private readonly ImageProcessor imageProcessor;
        private readonly UndoRedoService<BitmapSource> undoRedoService;
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

        // Constructor
        public MainViewModel()
        {
            undoRedoService = new UndoRedoService<BitmapSource>();
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

            loggingService.PropertyChanged += (s, e) => OnPropertyChanged(e.PropertyName);

            // Initialize Properties
            LogText = "";
            FileName = "No file loaded";
            ImageResolution = "N/A";
            ZoomLevel = "100%";
            ProcessingTime = "0 ms";
        }


        // Image Open-Save
        private void ExecuteOpenImage(object parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            openDialog.Filter = "Image files (*.png;*.jpeg;*.jpg;*.bmp)|*.png;*.jpeg;*.jpg;*.bmp|All files (*.*)|*.*";
            if (openDialog.ShowDialog() == true)
            {
                try
                {
                    double elapsedMs = ProcessTime.Measure(() =>
                    {
                        var bitmap = new BitmapImage();
                        bitmap.BeginInit();
                        bitmap.UriSource = new Uri(openDialog.FileName);
                        bitmap.CacheOption = BitmapCacheOption.OnLoad; // 메모리에 올리고 파일 잠금 해제
                        bitmap.EndInit();
                        bitmap.Freeze(); // UI 스레드 외에서 접근할 경우를 대비해 동결
                        BeforeImage = bitmap;
                        AfterImage = null;
                        FileName = Path.GetFileName(openDialog.FileName);
                        ImageResolution = $"{bitmap.PixelWidth}x{bitmap.PixelHeight}";

                        // Clear history for new image
                        undoRedoService.Clear();
                        historyService.Clear();
                        loggingService.AddLog($"Image loaded: {FileName}");
                        CommandManager.InvalidateRequerySuggested();
                    });
                    ProcessingTime = $"Load Time: {elapsedMs:F2} ms";

                }
                catch(Exception ex ) 
                {
                    MessageBox.Show($"이미지를 불러오는 중 오류가 발생했습니다.\n\n오류: {ex.Message}", 
                        "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void ExecuteSaveImage(object parameter)
        {
            if (AfterImage == null) return;

            SaveFileDialog saveDialog = new SaveFileDialog();
            saveDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg|BMP Image|*.bmp";

            if (saveDialog.ShowDialog() == true)
            { 
                try
                {
                    BitmapEncoder? encoder = null;
                    string extension = Path.GetExtension(saveDialog.FileName).ToLower();

                    switch (extension)
                    {
                        case ".png":
                            encoder = new PngBitmapEncoder();
                            break;
                        case ".jpg":
                        case ".jpeg":
                            encoder = new JpegBitmapEncoder();
                            break;
                        case ".bmp":
                            encoder = new BmpBitmapEncoder();
                            break;
                    }

                    if (encoder != null)
                    {
                        encoder.Frames.Add(BitmapFrame.Create(AfterImage));
                        using (var fileStream = new FileStream(saveDialog.FileName, FileMode.Create)) // 파일스트림을 직접 저장
                        {
                            encoder.Save(fileStream);
                            loggingService.AddLog($"Image saved: {Path.GetFileName(saveDialog.FileName)}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지를 저장하는 중 오류가 발생했습니다.\n\n오류: {ex.Message}",
                     "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }

            }
        }

        // 모든 필터 적용 메서드를 하나의 헬퍼 메서드로 통합하여 중복 제거
        private void ApplyProcessing(string historyMessage, Func<BitmapSource, BitmapSource> processFunction)
        {
            if (BeforeImage == null) return;
            try
            {
                // 1. Undo를 위해 현재 상태를 저장
                undoRedoService.AddState(BeforeImage);

                // 2. 시간 측정 및 이미지 처리
                var (processedImage, elapsedMs) = ProcessTime.Measure(() => processFunction(BeforeImage));

                // 3. 결과 업데이트
                AfterImage = processedImage;
                ProcessingTime = $"{historyMessage}: {elapsedMs:F2} ms";

                // 4. 히스토리 및 로그 기록 (서비스에 위임)
                historyService.AddHistory(historyMessage, elapsedMs);
                loggingService.AddLog($"{historyMessage} applied.");

                // CommandManager.InvalidateRequerySuggested()는 RelayCommand가 자동으로 처리해줌
            }
            catch (Exception ex)
            {
                MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
            
        }

        // 각 필터 Command는 새로 만든 헬퍼 메서드를 호출하기만 함
        // Color | Contrast
        private void ExecuteColorContrast(object parameter)
        {
            string processCommand = parameter.ToString();
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

        private void ExecuteFilterring(object parameter)
        {
            string processCommand = parameter.ToString();
            switch (processCommand)
            {
                case "Diff":
                    ApplyProcessing("Differentail", image => imageProcessor.ApplyDifferential(image));
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
                //case "FFT":
                //    ApplyProcessing("Laplacian", image => imageProcessor.ApplyFFT(image));
                //    break;
            }

        }
        private void ExecuteMorphorogy(object parameter)
        {
            string processCommand = parameter.ToString();
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
        private void ExecuteImageMatching(object parameter)
        {
            string processCommand = parameter.ToString();
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
        

        // Undo the last process
        private void ExecuteUndo(object parameter)
        {
            var previousState = undoRedoService.Undo(AfterImage);
            if (previousState != null) AfterImage = previousState;
        }

        private void ExecuteRedo(object parameter)
        {
            var nextState = undoRedoService.Redo(AfterImage);
            if (nextState != null) AfterImage = nextState;
        }
    }
}
