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
        private BitmapSource currentImage;
        public BitmapSource CurrentImage
        {
            get => currentImage;
            set
            {
                currentImage = value;
                OnPropertyChanged();
            }
        }

        private readonly ImageDocument imageDocument;

        // ViewModel에서 사용할 Command 정의
        public ICommand OpenImageCommand { get; }
        public ICommand SaveImageCommand { get; }
        public ICommand BinarizeCommand { get; }
        public ICommand OpenPreviewCommand { get; }

        public MainViewModel()
        {
            imageDocument = new ImageDocument();

            // Command와 실행 메서드 연결
            OpenImageCommand = new RelayCommand(ExecuteOpenImage);
            SaveImageCommand = new RelayCommand(ExecuteSaveImage);
            BinarizeCommand = new RelayCommand(ExecuteBinarize);
            OpenPreviewCommand = new RelayCommand(ExecuteOpenPreview);
        }

        private void ExecuteOpenImage(object parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            openDialog.Filter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png";

            if (openDialog.ShowDialog() == true)
            {
                CurrentImage = new BitmapImage(new Uri(openDialog.FileName));
            }
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
            // CurrentImage = _imageProcessor.ApplyBinarization(CurrentImage, 128);
            // _previewWindow.SetImage(CurrentImage);
        }

        private void ExecuteOpenPreview(object parameter)
        {

        }
    }
}
