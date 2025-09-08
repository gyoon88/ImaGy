using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ApplyImageMatchingCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly ImageProcessingService _imageProcessingService;

        public ApplyImageMatchingCommand(MainViewModel mainViewModel, ImageProcessingService imageProcessingService)
        {
            _mainViewModel = mainViewModel;
            _imageProcessingService = imageProcessingService;
        }

        public event EventHandler? CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object? parameter)
        {
            return !_mainViewModel.IsProcessing && _mainViewModel.AfterImage != null && _mainViewModel.TemplateImage != null;
        }

        public async void Execute(object? parameter)
        {
            if (parameter is string command)
            {
                _mainViewModel.IsProcessing = true;

                var processAction = _imageProcessingService.GetProcessAction(command, _mainViewModel);
                var (processedImage, elapsedMs) = await _imageProcessingService.ApplyProcessing(command, processAction, _mainViewModel.AfterImage);

                if (processedImage != null)
                {
                    _mainViewModel.AfterImage = processedImage;
                    _mainViewModel.ProcessingTime = $"{command}: {elapsedMs:F2} ms";
                }

                _mainViewModel.IsProcessing = false;
            }
        }
    }
}