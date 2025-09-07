using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class SaveImageCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly FileService _fileService;
        private readonly LoggingService _loggingService;

        public SaveImageCommand(MainViewModel mainViewModel, FileService fileService, LoggingService loggingService)
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
            return _mainViewModel.AfterImage != null;
        }

        public void Execute(object? parameter)
        {
            if (_mainViewModel.AfterImage != null)
            {
                _fileService.SaveImage(_mainViewModel.AfterImage);
                _loggingService.AddLog("Image saved."); // The FileService doesn't return the file name, so a generic message is used.
            }
        }
    }
}