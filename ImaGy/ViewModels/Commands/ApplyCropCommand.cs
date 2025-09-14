using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ApplyCropCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;

        public ApplyCropCommand(MainViewModel viewModel)
        {
            _mainViewModel = viewModel;
            _mainViewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(MainViewModel.IsProcessing) || e.PropertyName == nameof(MainViewModel.BeforeImage))
                {
                    CanExecuteChanged?.Invoke(this, System.EventArgs.Empty);
                }
            };
        }

        public event System.EventHandler? CanExecuteChanged;

        public bool CanExecute(object? parameter)
        {
            return !_mainViewModel.IsProcessing && _mainViewModel.BeforeImage != null;
        }

        public void Execute(object? parameter)
        {
            _mainViewModel.InteractionService.ToggleCropMode();
        }
    }
}
