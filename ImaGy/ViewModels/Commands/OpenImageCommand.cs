using ImaGy.Models;
using ImaGy.Services;
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class OpenImageCommand : System.Windows.Input.ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly FileService _fileService;
        private readonly LoggingService _loggingService;

        public OpenImageCommand(MainViewModel mainViewModel, FileService fileService, LoggingService loggingService)
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

        public async void Execute(object? parameter)
        {
            if (_mainViewModel.IsProcessing) return;

            try
            {
                _mainViewModel.IsProcessing = true;
                var result = await _fileService.OpenImage();
                if (result != null)
                {
                    _mainViewModel.BeforeImage = result.Bitmap;
                    _mainViewModel.AfterImage = result.Bitmap;
                    _mainViewModel.FileName = result.FileName;
                    _mainViewModel.ImageResolution = result.Resolution;
                    _mainViewModel.ProcessingTime = $"Load Time: {result.LoadTime:F2} ms";

                    // _mainViewModel.ImageLoadedCallback?.Invoke(result.Bitmap); // This needs to be handled

                    _mainViewModel.undoRedoService.Clear();
                    _mainViewModel.historyService.Clear();
                    _loggingService.AddLog($"Image loaded: {_mainViewModel.FileName}");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to load the image.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                _loggingService.AddLog($"Error loading image: {ex.Message}");
            }
            finally
            {
                _mainViewModel.IsProcessing = false;
            }
        }
    }
}