using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class OpenTemplateImageCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly FileService _fileService;
        private readonly LoggingService _loggingService;

        public OpenTemplateImageCommand(MainViewModel mainViewModel, FileService fileService, LoggingService loggingService)
        {
            _mainViewModel = mainViewModel;
            _fileService = fileService;
            _loggingService = loggingService;
        }

        public event EventHandler? CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object? parameter)
        {
            return !_mainViewModel.IsProcessing;
        }

        public void Execute(object? parameter)
        {
            var templateImage = _fileService.OpenTemplateImage();
            if (templateImage != null)
            {
                _mainViewModel.TemplateImage = templateImage;
                _loggingService.AddLog("Template image loaded.");
                // The logic to show the TemplateImageViewer is still in MainViewModel.
                // This should be moved to a navigation service or window service.
                // For now, I will leave it as it is and call a method on MainViewModel.
                _mainViewModel.ShowTemplateImageViewer();
            }
        }
    }
}