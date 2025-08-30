using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using Microsoft.Win32;
using System.IO;
using ImaGy.Model;
using System.Collections.ObjectModel;

namespace ImaGy.ViewModel
{
    internal class MainViewModel : BaseViewModel
    {
        // Fields
        private BitmapSource leftCurrentImage;
        private BitmapSource rightCurrentImage;
        private string fileName;
        private string imageResolution;
        private string zoomLevel;
        private string processingTime;
        private ObservableCollection<string> historyItems;
        private string logText;

        private Stack<BitmapSource> undoStack;
        private Stack<BitmapSource> redoStack;

        // Properties
        public BitmapSource LeftCurrentImage
        {
            get => leftCurrentImage;
            set { leftCurrentImage = value; OnPropertyChanged(); }
        }
        public BitmapSource RightCurrentImage
        {
            get => rightCurrentImage;
            set { rightCurrentImage = value; OnPropertyChanged(); }
        }
        public string FileName
        {
            get => fileName;
            set { fileName = value; OnPropertyChanged(); }
        }
        public string ImageResolution
        {
            get => imageResolution;
            set { imageResolution = value; OnPropertyChanged(); }
        }
        public string ZoomLevel
        {
            get => zoomLevel;
            set { zoomLevel = value; OnPropertyChanged(); }
        }
        public string ProcessingTime
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
        private readonly ImageDocument imageDocument;
        private readonly HistoryManager historyManager;
        private readonly ImageProcessor imageProcessor;

        // Commands
        public ICommand OpenImageCommand { get; }
        public ICommand SaveImageCommand { get; }
        public ICommand UndoCommand { get; }
        public ICommand RedoCommand { get; }
        public ICommand BinarizeCommand { get; }

        // Constructor
        public MainViewModel()
        {
            imageDocument = new ImageDocument();
            historyManager = new HistoryManager();
            imageProcessor = new ImageProcessor();

            undoStack = new Stack<BitmapSource>();
            redoStack = new Stack<BitmapSource>();

            // Commands
            OpenImageCommand = new RelayCommand(ExecuteOpenImage);
            SaveImageCommand = new RelayCommand(ExecuteSaveImage, _ => RightCurrentImage != null);
            UndoCommand = new RelayCommand(ExecuteUndo, _ => undoStack.Count > 0);
            RedoCommand = new RelayCommand(ExecuteRedo, _ => redoStack.Count > 0);
            BinarizeCommand = new RelayCommand(ExecuteBinarize, _ => RightCurrentImage != null);

            // Initialize Properties
            HistoryItems = new ObservableCollection<string>();
            LogText = "";
            FileName = "No file loaded";
            ImageResolution = "N/A";
            ZoomLevel = "100%";
            ProcessingTime = "0 ms";
        }

        private void ExecuteOpenImage(object parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            openDialog.Filter = "Image files (*.png;*.jpeg;*.jpg;*.bmp)|*.png;*.jpeg;*.jpg;*.bmp|All files (*.*)|*.*";
            if (openDialog.ShowDialog() == true)
            {
                var bitmap = new BitmapImage(new Uri(openDialog.FileName));
                LeftCurrentImage = bitmap;
                RightCurrentImage = bitmap; 
                FileName = Path.GetFileName(openDialog.FileName);
                ImageResolution = $"{bitmap.PixelWidth}x{bitmap.PixelHeight}";
                
                // Clear history for new image
                undoStack.Clear();
                redoStack.Clear();
                HistoryItems.Clear();
                CommandManager.InvalidateRequerySuggested();

                AddLog($"Image loaded: {FileName}");
            }
        }

        private void ExecuteSaveImage(object parameter)
        {
            if (RightCurrentImage == null) return;

            SaveFileDialog saveDialog = new SaveFileDialog();
            saveDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg|BMP Image|*.bmp";

            if (saveDialog.ShowDialog() == true)
            {
                BitmapEncoder encoder = null;
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
                    encoder.Frames.Add(BitmapFrame.Create(RightCurrentImage));
                    using (var fileStream = new FileStream(saveDialog.FileName, FileMode.Create))
                    {
                        encoder.Save(fileStream);
                        AddLog($"Image saved: {Path.GetFileName(saveDialog.FileName)}");
                    }
                }
            }
        }

        private void ExecuteBinarize(object parameter)
        {
            if (RightCurrentImage == null) return;

            undoStack.Push(RightCurrentImage);
            redoStack.Clear(); // Clear redo stack on new action

            RightCurrentImage = imageProcessor.ApplyBinarization(RightCurrentImage, 128);
            AddHistory("Binarization applied (threshold: 128)");
            CommandManager.InvalidateRequerySuggested();
        }

        // Undo the last process
        private void ExecuteUndo(object parameter)
        {
            if (undoStack.Count > 0)
            {
                redoStack.Push(RightCurrentImage);
                RightCurrentImage = undoStack.Pop();
                AddLog("Undo executed");
                HistoryItems.RemoveAt(0); // A simple way to remove last history item
                CommandManager.InvalidateRequerySuggested();
            }
        }

        private void ExecuteRedo(object parameter)
        {
            if (redoStack.Count > 0)
            {
                undoStack.Push(RightCurrentImage);
                RightCurrentImage = redoStack.Pop();
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
