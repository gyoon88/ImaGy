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

namespace ImaGy.ViewModel
{
    internal class MainViewModel : BaseViewModel
    {
        // Field
        private BitmapSource leftCurrentImage;
        private BitmapSource rightCurrentImage;

        // Property
        public BitmapSource LeftCurrentImage
        {
            get => leftCurrentImage;
            set
            {
                leftCurrentImage = value;
                OnPropertyChanged();
            }
        }
        public BitmapSource RightCurrentImage
        {
            get => rightCurrentImage;
            set
            {
                rightCurrentImage = value;
                OnPropertyChanged();
            }
        }

        // Model Class Imstance
        private readonly ImageDocument imageDocument;
        private readonly HistoryManager historyManager;

        // MainViewModel에서 사용할 Command 정의
        public ICommand ImageDocumentCommand { get; }

        public ICommand ImangeTransferCommand { get; }
        public ICommand BinarizeCommand { get; }
        public ICommand OpenPreviewCommand { get; }

        // Constructor
        public MainViewModel()
        {
            imageDocument = new ImageDocument();
            historyManager = new HistoryManager();

            // Command와 실행 메서드 연결
            ImageDocumentCommand = new RelayCommand(ExecuteOpenImage);
            ImangeTransferCommand = new RelayCommand(ExecuteSaveImage);

            BinarizeCommand = new RelayCommand(ExecuteBinarize);
            OpenPreviewCommand = new RelayCommand(ExecuteOpenPreview);
        }

        private void ExecuteOpenImage(object parameter)
        {
            string? command = parameter.ToString();
 

        }

        private void ExecuteSaveImage(object parameter)
        {
            if (CurrentImage == null) return;

            SaveFileDialog saveDialog = new SaveFileDialog();
            saveDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg";

            if (saveDialog.ShowDialog() == true)
            {
                BitmapEncoder encoder;
                if (saveDialog.FilterIndex == 1)
                {
                    encoder = new PngBitmapEncoder();
                }
                else
                {
                    encoder = new JpegBitmapEncoder();
                }

                encoder.Frames.Add(BitmapFrame.Create(CurrentImage));
                using (var fileStream = new FileStream(saveDialog.FileName, FileMode.Create))
                {
                    encoder.Save(fileStream);
                }
            }
        }

        private void ExecuteBinarize(object parameter)
        {
            if (CurrentImage == null) return;
            // ImageProcessor를 통해 이미지를 처리하고 결과 이미지를 CurrentImage에 할당
            // LeftCurrentImage = _imageProcessor.ApplyBinarization(LeftCurrentImage, 128);
            // _previewWindow.SetImage(LeftCurrentImage);
        }

        private void ExecuteOpenPreview(object parameter)
        {

        }
    }
}
