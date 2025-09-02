using System.Linq;
using System.Windows.Media;

namespace ImaGy.ViewModel
{
    public class HistogramViewModel : BaseViewModel
    {
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

        public HistogramViewModel(int[] data)
        {
            HistogramData = data;
            MaxHistogramValue = data.Any() ? data.Max() : 0;
        }
    }
}