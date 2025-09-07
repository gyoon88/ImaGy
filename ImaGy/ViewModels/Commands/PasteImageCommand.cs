using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class PasteImageCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly ClipboardImageService _clipboardService;

        public PasteImageCommand(MainViewModel mainViewModel, ClipboardImageService clipboardService)
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
            return true; // Can always try to paste
        }

        public void Execute(object? parameter)
        {
            var image = _clipboardService.GetImage();
            if (image != null)
            {
                _mainViewModel.AfterImage = image;
                _mainViewModel.BeforeImage = image;
            }
        }
    }
}