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
        private readonly ImageProcessor _imageProcessor;
        private readonly ImageProcessorSSE _imageProcessorSse;

        private readonly ColorContrastProcess _colorContrastProcessor;
        private readonly MatchingProcessor _matchingProcessor;
        private readonly FilterProcessor _filterProcessor;
        private readonly MorphologyProcessor _morphologyProcessor;
        
        private readonly UndoRedoService<BitmapSource?> _undoRedoService;
        private readonly HistoryService _historyService;
        private readonly LoggingService _loggingService;

        public ImageProcessingService(
            ImageProcessor imageProcessor,
            ImageProcessorSSE imageProcessorSse,
            ColorContrastProcess colorContrastProcessor,
            MatchingProcessor matchingProcessor,
            FilterProcessor filterProcessor,
            MorphologyProcessor morphologyProcessor,


            UndoRedoService<BitmapSource?> undoRedoService,
            HistoryService historyService,
            LoggingService loggingService)
        {
            _imageProcessor = imageProcessor;
            _imageProcessorSse = imageProcessorSse;
            
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

            _undoRedoService.AddState(imageToProcess);
            var (processedImage, elapsedMs) = await Task.Run(() =>
            {
                return ProcessTime.Measure(() => processAction(imageToProcess));
            });

            _historyService.AddHistory(processName, elapsedMs);
            _loggingService.AddLog($"{processName} applied.");

            return (processedImage, elapsedMs);
        }

        public Func<BitmapSource, BitmapSource> GetProcessAction(string processCommand, BitmapSource template)
        {
            return processCommand switch
            {
                "NCC" => (image) => _matchingProcessor.ApplyNCC(image, template),
                "SAD" => (image) => _matchingProcessor.ApplySAD(image, template),
                "SSD" => (image) => _matchingProcessor.ApplySSD(image, template),
                _ => GetProcessAction(processCommand),
            };
        }

        public Func<BitmapSource, BitmapSource> GetProcessAction(string processCommand, double sigma, int kernelSize)
        {
            
            return processCommand switch
            {
                "Average" => (image) => _filterProcessor.ApplyAverageBlur(image, kernelSize),
                "Average_SSE" => (image) => _imageProcessorSse.ApplyAverageBlurSse(image, kernelSize),
                "Gaussian" => (image) => _filterProcessor.ApplyGaussianBlur(image, sigma, kernelSize),
                "Gaussian_SSE" => (image) => _imageProcessorSse.ApplyGaussianBlurSse(image, sigma, kernelSize),
                _ => throw new ArgumentException("Invalid process command", nameof(processCommand)),
            };
        }
        public Func<BitmapSource, BitmapSource> GetProcessAction(string processCommand)
        {
            
            return processCommand switch
            {
                "Bin" => (image) => _colorContrastProcessor.ApplyBinarization(image, 128),
                "Equal" => (image) => _colorContrastProcessor.ApplyEqualization(image),
                "Diff" => (image) => _filterProcessor.ApplyDifferential(image),
                "Diff_SSE" => (image) => _imageProcessorSse.ApplyDifferentialSse(image),
                "Sobel" => (image) => _filterProcessor.ApplySobel(image),
                "Sobel_SSE" => (image) => _imageProcessorSse.ApplySobelSse(image),
                "Laplace" => (image) => _filterProcessor.ApplyLaplacian(image),
                "Laplace_SSE" => (image) => _imageProcessorSse.ApplyLaplacianSse(image),

                "Dilation" => (image) => _morphologyProcessor.ApplyDilation(image),
                "Dilation_SSE" => (image) => _imageProcessorSse.ApplyDilationSse(image),
                "Erosion" => (image) => _morphologyProcessor.ApplyErosion(image),
                "Erosion_SSE" => (image) => _imageProcessorSse.ApplyErosionSse(image),
                _ => throw new ArgumentException("Invalid process command", nameof(processCommand)),
            };
        }
    }
}