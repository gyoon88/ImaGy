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
        public ObservableCollection<string> HistoryItems
        {
            get => historyItems;
            set { historyItems = value; OnPropertyChanged(); }
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

        // Commands
        public ICommand OpenImageCommand { get; }
        public ICommand SaveImageCommand { get; }
        public ICommand UndoCommand { get; }
        public ICommand RedoCommand { get; }
        public ICommand BinarizeCommand { get; }
        //public ICommand BinarizeCommand { get; }
        //public ICommand BinarizeCommand { get; }
        //public ICommand BinarizeCommand { get; }
        //public ICommand BinarizeCommand { get; }
        //public ICommand BinarizeCommand { get; }
        //public ICommand BinarizeCommand { get; }


        // Constructor
        public MainViewModel()
        {
            //imageDocument = new ImageDocument();
            //historyManager = new HistoryManager();
            imageProcessor = new ImageProcessor();

            undoStack = new Stack<BitmapSource>();
            redoStack = new Stack<BitmapSource>();

            // Commands
            OpenImageCommand = new RelayCommand(ExecuteOpenImage);
            SaveImageCommand = new RelayCommand(ExecuteSaveImage, _ => AfterImage != null);
            UndoCommand = new RelayCommand(ExecuteUndo, _ => undoStack.Count > 0);
            RedoCommand = new RelayCommand(ExecuteRedo, _ => redoStack.Count > 0);
            BinarizeCommand = new RelayCommand(ExecuteBinarize, _ => BeforeImage != null);

            // Initialize Properties
            HistoryItems = new ObservableCollection<string>();
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
                        undoStack.Clear();
                        redoStack.Clear();
                        HistoryItems.Clear();
                        CommandManager.InvalidateRequerySuggested();
                    });
                    ProcessingTime = $"Load Time: {elapsedMs:F2} ms";
                    AddLog($"Image loaded: {FileName} ({elapsedMs:F2} ms)");
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
                        AddLog($"Image saved: {Path.GetFileName(saveDialog.FileName)}");
                    }
                }
            }
        }

        // Image Processing ExecuteMethod
        private void ExecuteBinarize(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyBinarization(BeforeImage, 128);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Binarization: {elapsedMs:F2} ms";
                    AddHistory($"Binarization applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }

        private void ExecuteEqualize(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyEqualization(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Equalization: {elapsedMs:F2} ms";
                    AddHistory($"Equalization applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteDifferential(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyDifferential(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Differential: {elapsedMs:F2} ms";
                    AddHistory($"Differential applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteSobel(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplySobel(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Sobel: {elapsedMs:F2} ms";
                    AddHistory($"Sobel applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteLaplacian(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyLaplacian(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Laplacian: {elapsedMs:F2} ms";
                    AddHistory($"Laplacian applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteAverageBlur(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyAverageBlur(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"AverageBlur: {elapsedMs:F2} ms";
                    AddHistory($"AverageBlur applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteGaussianBlur(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyGaussianBlur(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"GaussianBlur: {elapsedMs:F2} ms";
                    AddHistory($"GaussianBlur applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteDilation(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyDilation(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Dilation: {elapsedMs:F2} ms";
                    AddHistory($"Dilation applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteErosion(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyErosion(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"Erosion: {elapsedMs:F2} ms";
                    AddHistory($"Erosion applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }
        private void ExecuteNCC(object parameter)
        {
            if (BeforeImage != null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplyNCC(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"NCC: {elapsedMs:F2} ms";
                    AddHistory($"NCC applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류",
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }
        private void ExecuteSAD(object parameter)
        {
            if (BeforeImage != null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplySAD(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"SAD: {elapsedMs:F2} ms";
                    AddHistory($"SAD applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류",
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }
        private void ExecuteSSD(object parameter)
        {
            if (BeforeImage !=null)
            {
                try
                {
                    int threshold = 128;
                    undoStack.Push(BeforeImage);
                    redoStack.Clear(); // Clear redo stack on new action
                    var (processedImage, elapsedMs) = ProcessTime.Measure(() =>
                    {// tuple 형태로 반환
                        return imageProcessor.ApplySSD(BeforeImage);
                    });
                    AfterImage = processedImage;
                    ProcessingTime = $"SSD: {elapsedMs:F2} ms";
                    AddHistory($"SSD applied (threshold: {threshold})\n{ProcessingTime}");
                    CommandManager.InvalidateRequerySuggested();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 처리 중 오류가 발생했습니다.\n\n오류: {ex.Message}", "오류", 
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }            
        }

        // Undo the last process
        private void ExecuteUndo(object parameter)
        {
            if (undoStack.Count > 0)
            {
                redoStack.Push(AfterImage);
                AfterImage = undoStack.Pop();
                AddLog("Undo executed");
                HistoryItems.RemoveAt(0); // A simple way to remove last history item
                CommandManager.InvalidateRequerySuggested();
            }
        }

        private void ExecuteRedo(object parameter)
        {
            if (redoStack.Count > 0)
            {
                undoStack.Push(AfterImage);
                AfterImage = redoStack.Pop();
                AddLog("Redo executed");
                // This is tricky, we need to get the history message back
                // For now, let's just add a generic message
                AddHistory("Redo action"); 
                CommandManager.InvalidateRequerySuggested();
            }
        }

        private void AddHistory(string message)
        {
            string timestamp = DateTime.Now.ToString("[HH:mm:ss]");
            HistoryItems.Insert(0, $"{timestamp} {message}");
        }

        private void AddLog(string message)
        {
            string timestamp = DateTime.Now.ToString("[yyyy-MM-dd HH:mm:ss]");
            LogText += $"{timestamp} [INFO] {message}\n";
        }
    }
}
