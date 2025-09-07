using ImaGy.Services;
using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ViewHistogramCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly HistogramService _histogramService;

        public ViewHistogramCommand(MainViewModel mainViewModel, HistogramService histogramService)
        {
            _mainViewModel = mainViewModel;
            _histogramService = histogramService;
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
            _histogramService.ShowHistogram(_mainViewModel);
        }
    }
}