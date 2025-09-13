using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class ApplyCropCommand : ICommand
    {
        private readonly MainViewModel _viewModel;

        public ApplyCropCommand(MainViewModel viewModel)
        {
            _viewModel = viewModel;
            _viewModel.PropertyChanged += (s, e) =>
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
            return !_viewModel.IsProcessing && _viewModel.BeforeImage != null;
        }

        public void Execute(object? parameter)
        {
            // Toggle crop mode
            _viewModel.IsInCropMode = !_viewModel.IsInCropMode;
            // Reset any existing crop rectangle when entering the mode
            if (_viewModel.IsInCropMode)
            {
                _viewModel.CropRectangle = new System.Windows.Rect();
            }
        }
    }
}
