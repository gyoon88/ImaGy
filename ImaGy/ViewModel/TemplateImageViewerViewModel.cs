using System.Windows.Media.Imaging;
using ImaGy.Model;

namespace ImaGy.ViewModel
{
    public class TemplateImageViewerViewModel : BaseViewModel
    {
        private MainViewModel _mainViewModel;

        private BitmapSource? templateImage;
        public BitmapSource? TemplateImage
        {
            get => templateImage;
            set => SetProperty(ref templateImage, value);
        }

        public TemplateImageViewerViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            // Initialize TemplateImage with the current value from MainViewModel
            TemplateImage = _mainViewModel.TemplateImage;

            // Subscribe to PropertyChanged event to update if TemplateImage changes in MainViewModel
            _mainViewModel.PropertyChanged += MainViewModel_PropertyChanged;
        }

        private void MainViewModel_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.TemplateImage))
            {
                TemplateImage = _mainViewModel.TemplateImage;
            }
        }
    }
}
