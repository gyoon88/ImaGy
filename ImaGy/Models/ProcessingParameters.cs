namespace ImaGy.Models
{
    public class ProcessingParameters : ViewModels.BaseViewModel
    {
        private int threshold = 128;
        public int Threshold
        {
            get => threshold;
            set => SetProperty(ref threshold, value);
        }

        private int kernelSize = 3;
        public int KernelSize
        {
            get => kernelSize;
            set => SetProperty(ref kernelSize, value);
        }

        private double sigma = 1.0;
        public double Sigma
        {
            get => sigma;
            set => SetProperty(ref sigma, value);
        }

        private bool useCircularKernel = false;
        public bool UseCircularKernel
        {
            get => useCircularKernel;
            set => SetProperty(ref useCircularKernel, value);
        }
    }
}
