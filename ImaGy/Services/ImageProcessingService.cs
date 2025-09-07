using ImaGy.Models;
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace ImaGy.Services
{
    public class ImageProcessingService
    {
        private readonly ImageProcessor _imageProcessor;
        private readonly ImageProcessorSSE _imageProcessorSse;
        private readonly UndoRedoService<BitmapSource?> _undoRedoService;
        private readonly HistoryService _historyService;
        private readonly LoggingService _loggingService;

        public ImageProcessingService(
            ImageProcessor imageProcessor,
            ImageProcessorSSE imageProcessorSse,
            UndoRedoService<BitmapSource?> undoRedoService,
            HistoryService historyService,
            LoggingService loggingService)
        {
            _imageProcessor = imageProcessor;
            _imageProcessorSse = imageProcessorSse;
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
                "NCC" => (image) => _imageProcessor.ApplyNCC(image, template),
                "SAD" => (image) => _imageProcessor.ApplySAD(image, template),
                "SSD" => (image) => _imageProcessor.ApplySSD(image, template),
                _ => GetProcessAction(processCommand),
            };
        }

        public Func<BitmapSource, BitmapSource> GetProcessAction(string processCommand)
        {
            return processCommand switch
            {
                "Bin" => (image) => _imageProcessor.ApplyBinarization(image, 128),
                "Equal" => (image) => _imageProcessor.ApplyEqualization(image),
                "Diff" => (image) => _imageProcessor.ApplyDifferential(image),
                "Diff_SSE" => (image) => _imageProcessorSse.ApplyDifferentialSse(image),
                "Sobel" => (image) => _imageProcessor.ApplySobel(image),
                "Sobel_SSE" => (image) => _imageProcessorSse.ApplySobelSse(image),
                "Laplace" => (image) => _imageProcessor.ApplyLaplacian(image),
                "Laplace_SSE" => (image) => _imageProcessorSse.ApplyLaplacianSse(image),
                "Average" => (image) => _imageProcessor.ApplyAverageBlur(image),
                "Average_SSE" => (image) => _imageProcessorSse.ApplyAverageBlurSse(image),
                "Gaussian" => (image) => _imageProcessor.ApplyGaussianBlur(image),
                "Gaussian_SSE" => (image) => _imageProcessorSse.ApplyGaussianBlurSse(image),
                "Dilation" => (image) => _imageProcessor.ApplyDilation(image),
                "Dilation_SSE" => (image) => _imageProcessorSse.ApplyDilationSse(image),
                "Erosion" => (image) => _imageProcessor.ApplyErosion(image),
                "Erosion_SSE" => (image) => _imageProcessorSse.ApplyErosionSse(image),
                _ => throw new ArgumentException("Invalid process command", nameof(processCommand)),
            };
        }
    }
}