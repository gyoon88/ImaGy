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
                        bitmap.CacheOption = BitmapCacheOption.OnLoad; // �޸𸮿� �ø��� ���� ��� ����
                        bitmap.EndInit();
                        bitmap.Freeze(); // UI ������ �ܿ��� ������ ��츦 ����� ����
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
                    MessageBox.Show($"�̹����� �ҷ����� �� ������ �߻��߽��ϴ�.\n\n����: {ex.Message}", 
                        "����", MessageBoxButton.OK, MessageBoxImage.Error);
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
                        using (var fileStream = new FileStream(saveDialog.FileName, FileMode.Create)) // ���Ͻ�Ʈ���� ���� ����
                        {
                            encoder.Save(fileStream);
                            loggingService.AddLog($"Image saved: {Path.GetFileName(saveDialog.FileName)}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"�̹����� �����ϴ� �� ������ �߻��߽��ϴ�.\n\n����: {ex.Message}",
                     "����", MessageBoxButton.OK, MessageBoxImage.Error);
                }

            }
        }

        // ��� ���� ���� �޼��带 �ϳ��� ���� �޼���� �����Ͽ� �ߺ� ����
        private void ApplyProcessing(string historyMessage, Func<BitmapSource, BitmapSource> processFunction)
        {
            if (BeforeImage == null) return;
            try
            {
                // 1. Undo�� ���� ���� ���¸� ����
                undoRedoService.AddState(BeforeImage);

                // 2. �ð� ���� �� �̹��� ó��
                var (processedImage, elapsedMs) = ProcessTime.Measure(() => processFunction(BeforeImage));

                // 3. ��� ������Ʈ
                AfterImage = processedImage;
                ProcessingTime = $"{historyMessage}: {elapsedMs:F2} ms";

                // 4. �����丮 �� �α� ��� (���񽺿� ����)
                historyService.AddHistory(historyMessage, elapsedMs);
                loggingService.AddLog($"{historyMessage} applied.");

                // CommandManager.InvalidateRequerySuggested()�� RelayCommand�� �ڵ����� ó������
            }
            catch (Exception ex)
            {
                MessageBox.Show($"�̹��� ó�� �� ������ �߻��߽��ϴ�.\n\n����: {ex.Message}", "����",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
            
        }

        // �� ���� Command�� ���� ���� ���� �޼��带 ȣ���ϱ⸸ ��
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
