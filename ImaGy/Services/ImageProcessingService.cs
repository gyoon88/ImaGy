using ImaGy.Models;
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;

namespace ImaGy.Services
{
    public class ImageProcessingService
    {
        private readonly ColorContrastProcess _colorContrastProcessor;
        private readonly MatchingProcessor _matchingProcessor;
        private readonly FilterProcessor _filterProcessor;
        private readonly MorphologyProcessor _morphologyProcessor;

        private readonly UndoRedoService<BitmapSource?> _undoRedoService;
        private readonly HistoryService _historyService;
        private readonly LoggingService _loggingService;

        // Constructor
        public ImageProcessingService(
            ColorContrastProcess colorContrastProcessor,
            MatchingProcessor matchingProcessor,
            FilterProcessor filterProcessor,
            MorphologyProcessor morphologyProcessor,
            UndoRedoService<BitmapSource?> undoRedoService,
            HistoryService historyService,
            LoggingService loggingService
            )
        {
            _colorContrastProcessor = colorContrastProcessor;
            _matchingProcessor = matchingProcessor;
            _filterProcessor = filterProcessor;
            _morphologyProcessor = morphologyProcessor;
            _undoRedoService = undoRedoService;
            _historyService = historyService;
            _loggingService = loggingService;
        }

        public async Task<(BitmapSource? processedImage, double elapsedMs)> ApplyProcessing(
            string processName,
            Func<BitmapSource, BitmapSource> processAction,
            BitmapSource? imageToProcess)
        {
            if (imageToProcess == null)
            {
                _loggingService.AddLog("No image loaded for processing.");
                MessageBox.Show("Please load an image first.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return (null, 0);
            }
            if (processAction == null)
            {
                throw new ArgumentException($"Invalid or unsupported filter type: {processName}");
            }
            _undoRedoService.AddState(imageToProcess);
            var (processedImage, elapsedMs) = await Task.Run(() =>
            {
                return ProcessTime.Measure(() => processAction(imageToProcess));
            });

            _historyService.AddHistory(processName, elapsedMs);
            _loggingService.AddLog($"{processName} applied.");

            return (processedImage, elapsedMs);
        }


        public Func<BitmapSource, BitmapSource> GetProcessAction(string processCommand, ViewModels.MainViewModel vm)
        {
            return processCommand switch
            {
                // Color & Contrast
                "Bin" => (image) => _colorContrastProcessor.ApplyBinarization(image, vm.Parameters.Threshold),
                "Bright" => (image) => _colorContrastProcessor.ApplyBinarization(image, vm.Parameters.Threshold),
                "Bin_Otsu" => (image) => _colorContrastProcessor.ApplyOtsuBinarization(image),
                "Clustering" => (image) => _colorContrastProcessor.ApplyKMeansClustering(image, vm.Parameters.KGroup, vm.Parameters.Iteration, false, vm.IsColor),
                "ClusteringLoc" => (image) => _colorContrastProcessor.ApplyKMeansClustering(image, vm.Parameters.KGroup, vm.Parameters.Iteration, true, vm.IsColor),

                "Gray" => (image) => _colorContrastProcessor.ToGrayscale(image),

                "Equal" => (image) => _colorContrastProcessor.ApplyEqualization(image),
                "Equal_color" => (image) => _colorContrastProcessor.ApplyColorEqualization(image), //

                // Filters
                "Diff" => (image) => _filterProcessor.ApplyDifferential(image, vm.IsColor),
                "Sobel" => (image) => _filterProcessor.ApplySobel(image, vm.Parameters.KernelSize, vm.IsColor),

                "Laplace" => (image) => _filterProcessor.ApplyLaplacian(image, vm.Parameters.KernelSize, vm.IsColor),

                "FFT" => (image) => _filterProcessor.ApplyFFT(image, vm.Parameters.KernelSize, vm.IsColor, vm.Parameters.IsInverse, vm.Parameters.IsCPU, vm.Parameters.IsPhase),
                "FFT_Low" => (image) => _filterProcessor.ApplyFrequencyFilter(image, 0, vm.Parameters.RadiusRatio),
                "FFT_High" => (image) => _filterProcessor.ApplyFrequencyFilter(image, 1, vm.Parameters.RadiusRatio),
                "FFT_Band" => (image) => _filterProcessor.ApplyAxialBandStopFilter(image, vm.Parameters.LowFreqRadius, vm.Parameters.MagnitudeThreshold),
                "Average" => (image) => _filterProcessor.ApplyAverageBlur(image, vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor),

                "Gaussian" => (image) => _filterProcessor.ApplyGaussianBlur(image, vm.Parameters.Sigma, vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor),

                // Morphology
                "Dilation" => (image) => _morphologyProcessor.ApplyDilation(image, vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor),
                "Erosion" => (image) => _morphologyProcessor.ApplyErosion(image, vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor),
                "Opening" => (image) => _morphologyProcessor.ApplyDilation(_morphologyProcessor.ApplyErosion(image, vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor), vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor),
                "Closing" => (image) => _morphologyProcessor.ApplyErosion(_morphologyProcessor.ApplyDilation(image, vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor), vm.Parameters.KernelSize, vm.Parameters.UseCircularKernel, vm.IsColor),

                // Matching - Template
                "NCC" => (image) => _matchingProcessor.ApplyNCC(image, vm.TemplateImage!),
                "SAD" => (image) => _matchingProcessor.ApplySAD(image, vm.TemplateImage!),
                "SSD" => (image) => _matchingProcessor.ApplySSD(image, vm.TemplateImage!),

                _ => null,
            };
        }
    }
}