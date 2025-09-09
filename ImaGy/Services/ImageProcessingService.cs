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
                "Bin" => (image) => _colorContrastProcessor.ApplyBinarization(image, vm.Threshold), 
                "Bin_Othu" => (image) => _colorContrastProcessor.ApplyOtsuBinarization(image),

                "Grey" => (image) => _colorContrastProcessor.ToGrayscale(image), 

                "Equal" => (image) => _colorContrastProcessor.ApplyEqualization(image),
                "Equal_color" => (image) => _colorContrastProcessor.ApplyColorEqualization(image),

                // Filters
                "Average" => (image) => _filterProcessor.ApplyAverageBlur(image, vm.KernelSize),
                "Average_SSE" => (image) => _imageProcessorSse.ApplyAverageBlurSse(image, vm.KernelSize),
                "Gaussian" => (image) => _filterProcessor.ApplyGaussianBlur(image, vm.Sigma, vm.KernelSize),
                "Gaussian_SSE" => (image) => _imageProcessorSse.ApplyGaussianBlurSse(image, vm.Sigma, vm.KernelSize),
                "Diff" => (image) => _filterProcessor.ApplyDifferential(image),
                "Diff_SSE" => (image) => _imageProcessorSse.ApplyDifferentialSse(image),
                "Sobel" => (image) => _filterProcessor.ApplySobel(image, vm.KernelSize),
                "Sobel_SSE" => (image) => _imageProcessorSse.ApplySobelSse(image),
                "Laplace" => (image) => _filterProcessor.ApplyLaplacian(image, vm.KernelSize),
                "Laplace_SSE" => (image) => _imageProcessorSse.ApplyLaplacianSse(image),

                // Morphology
                "Dilation" => (image) => _morphologyProcessor.ApplyDilation(image, vm.KernelSize),
                "Dilation_SSE" => (image) => _imageProcessorSse.ApplyDilationSse(image),
                "Erosion" => (image) => _morphologyProcessor.ApplyErosion(image, vm.KernelSize),
                "Erosion_SSE" => (image) => _imageProcessorSse.ApplyErosionSse(image),

                // Matching - Template 이미지가 필요
                "NCC" => (image) => _matchingProcessor.ApplyNCC(image, vm.TemplateImage),
                "SAD" => (image) => _matchingProcessor.ApplySAD(image, vm.TemplateImage),
                "SSD" => (image) => _matchingProcessor.ApplySSD(image, vm.TemplateImage),

                _ => null, // 잘못된 명령에 대해 null 반환하여 ApplyProcessing에서 처리
            };
        }
    }
}