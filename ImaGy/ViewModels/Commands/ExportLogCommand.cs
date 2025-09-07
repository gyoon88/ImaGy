using ImaGy.Services;
using System;
using System.Windows;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ExportLogCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly LoggingService _loggingService;
        private readonly FileService _fileService;

        public ExportLogCommand(MainViewModel mainViewModel, LoggingService loggingService, FileService fileService)
        {
            _mainViewModel = mainViewModel;
            _loggingService = loggingService;
            _fileService = fileService;
        }

        public event EventHandler? CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object? parameter)
        {
            return !string.IsNullOrEmpty(_mainViewModel.LogText);
        }

        public void Execute(object? parameter)
        {
            var filePath = _fileService.GetSaveFilePath("Log.csv");
            if (filePath != null)
            {
                try
                {
                    _loggingService.ExportToCsv(filePath);
                    _loggingService.AddLog($"Log exported to: {filePath}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to export log.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    _loggingService.AddLog($"Error exporting log: {ex.Message}");
                }
            }
        }
    }
}