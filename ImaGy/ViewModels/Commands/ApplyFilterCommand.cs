using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ApplyFilterCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly ImageProcessingService _imageProcessingService;

        public ApplyFilterCommand(MainViewModel mainViewModel, ImageProcessingService imageProcessingService)
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
            return !_mainViewModel.IsProcessing && _mainViewModel.AfterImage != null;
        }

        public async void Execute(object? parameter)
        {
            if (parameter is string filterType)
            {
                _mainViewModel.IsProcessing = true;
                
                var processAction = _imageProcessingService.GetProcessAction(filterType, _mainViewModel); // get Process
                var (processedImage, elapsedMs) = await _imageProcessingService.ApplyProcessing(filterType, processAction, _mainViewModel.AfterImage); // 위에서 정해진 프로세스
                
                if(processedImage != null)
                {
                    _mainViewModel.AfterImage = processedImage;
                    _mainViewModel.ProcessingTime = $"{filterType}: {elapsedMs:F2} ms";
                }

                _mainViewModel.IsProcessing = false;
            }
        }
    }
}