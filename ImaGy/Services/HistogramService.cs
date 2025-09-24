using ImaGy.ViewModels;
using ImaGy.View;

namespace ImaGy.Services
{
    /// <summary>
    /// 
    /// </summary>
    public class HistogramService
    {
        public void ShowHistogram(MainViewModel mainViewModel)
        {
            HistogramViewModel histogramViewModel = new HistogramViewModel(mainViewModel);
            HistogramWindow histogramWindow = new HistogramWindow
            {
                DataContext = histogramViewModel
            };
            histogramWindow.Show();
        }
    }
}