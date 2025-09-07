using ImaGy.Services;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ExportHistoryCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly HistoryService _historyService;
        private readonly LoggingService _loggingService;
        private readonly FileService _fileService;

        public ExportHistoryCommand(MainViewModel mainViewModel, HistoryService historyService, LoggingService loggingService, FileService fileService)
        {
            _mainViewModel = mainViewModel;
            _historyService = historyService;
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
            return _mainViewModel.HistoryItems.Any();
        }

        public void Execute(object? parameter)
        {
            var filePath = _fileService.GetSaveFilePath("History.csv");
            if (filePath != null)
            {
                try
                {
                    _historyService.ExportToCsv(filePath);
                    _loggingService.AddLog($"History exported to: {filePath}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to export history.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    _loggingService.AddLog($"Error exporting history: {ex.Message}");
                }
            }
        }
    }
}