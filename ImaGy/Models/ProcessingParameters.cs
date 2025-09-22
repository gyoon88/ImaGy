namespace ImaGy.Models
{
    public class ProcessingParameters : ViewModels.BaseViewModel
    {
        private int threshold = 128;
        private int kernelSize = 3;
        private double sigma = 1.0;
        
        // Toggle switch 
        private bool useCircularKernel = false;
        private bool isCPU = false;
        private bool isInverse = false;
        private bool isPhase = false;
        // K-Means Clustering
        private int kGroup = 2;
        private int iteration = 20;

        // FFT row-high pass Filter
        private double radiusRatio = 0.5;


        public int Threshold
        {
            get => threshold;
            set => SetProperty(ref threshold, value);
        }
        public int KernelSize
        {
            get => kernelSize;
            set => SetProperty(ref kernelSize, value);
        }
        public double Sigma
        {
            get => sigma;
            set => SetProperty(ref sigma, value);
        }
        public bool UseCircularKernel
        {
            get => useCircularKernel;
            set => SetProperty(ref useCircularKernel, value);
        }
        public bool IsCPU
        {
            get => isCPU;
            set => SetProperty(ref isCPU, value);
        }
        public bool IsInverse
        {
            get => isInverse;
            set => SetProperty(ref isInverse, value);
        }
        public bool IsPhase
        {
            get => isPhase;
            set => SetProperty(ref isPhase, value);
        }
        public int KGroup
        {
            get => kGroup;
            set => SetProperty(ref kGroup, value);
        }
        public int Iteration
        {
            get => iteration;
            set => SetProperty(ref iteration, value);
        }        
        public double RadiusRatio
        {
            get => radiusRatio;
            set => SetProperty(ref radiusRatio, value);
        }
    }
}
