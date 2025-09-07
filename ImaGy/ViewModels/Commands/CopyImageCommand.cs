using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class CopyImageCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly ClipboardImageService _clipboardService;

        public CopyImageCommand(MainViewModel mainViewModel, ClipboardImageService clipboardService)
        {
            _mainViewModel = mainViewModel;
            _clipboardService = clipboardService;
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
                _clipboardService.SetImage(_mainViewModel.AfterImage);
            }
        }
    }
}