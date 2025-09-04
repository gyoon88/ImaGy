using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ImaGy.Model;

namespace ImaGy.ViewModel
{
    public class HistogramViewModel : BaseViewModel
    {
        private MainViewModel mainViewModel;

        private int[]? histogramData;
        public int[]? HistogramData
        {
            get => histogramData;
            set => SetProperty(ref histogramData, value);
        }

        private int maxHistogramValue;
        public int MaxHistogramValue
        {
            get => maxHistogramValue;
            set => SetProperty(ref maxHistogramValue, value);
        }

        public HistogramViewModel(MainViewModel mainViewModel)
        {
            this.mainViewModel = mainViewModel;
            this.mainViewModel.PropertyChanged += MainViewModel_PropertyChanged;
            CalculateHistogram();
        }

        private void MainViewModel_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.AfterImage) || e.PropertyName == nameof(MainViewModel.BeforeImage))
            {
                CalculateHistogram();
            }
        }

        private void CalculateHistogram()
        {
            BitmapSource? imageSource = mainViewModel.AfterImage ?? mainViewModel.BeforeImage;
            if (imageSource != null)
            {
                HistogramData = ServeHistogram.CalculateGrayscaleHistogram(imageSource);
                MaxHistogramValue = HistogramData.Any() ? HistogramData.Max() : 0;
            }
            else
            {
                HistogramData = null;
                MaxHistogramValue = 0;
            }
        }
    }
}