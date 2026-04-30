using System;
using System.Windows;
using System.Windows.Input;
using ImaGy.Services;
using ImaGy.View;

namespace ImaGy.ViewModels.Commands
{
    public sealed class OpenImageEditCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly LoggingService _loggingService;

        public OpenImageEditCommand(MainViewModel mainViewModel, LoggingService loggingService)
        {
            _mainViewModel = mainViewModel;
            _loggingService = loggingService;
        }

        public event EventHandler? CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }

        public bool CanExecute(object? parameter) => _mainViewModel.AfterImage != null && !_mainViewModel.IsProcessing;

        public void Execute(object? parameter)
        {
            try
            {
                var main = System.Windows.Application.Current?.MainWindow;
                var win = new ImageEditWindow { Owner = main };
                if (main != null)
                    win.WindowStartupLocation = WindowStartupLocation.CenterOwner;
                var vm = new ImageEditViewModel(_mainViewModel, _loggingService);
                vm.RegisterCloseHandler(() => win.Close());
                win.DataContext = vm;
                win.Show();
                win.Activate();
            }
            catch (InvalidOperationException ex)
            {
                System.Windows.MessageBox.Show(ex.Message, "Image Edit", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
            }
        }
    }
}
