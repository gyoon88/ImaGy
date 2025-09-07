using ImaGy.Models;
using System.Windows.Input;
using System.Windows;

namespace ImaGy.ViewModels
{
    public class RoiViewModel : BaseViewModel
    {
        private RoiModel? _currentRoi;
        public RoiModel? CurrentRoi
        {
            get => _currentRoi;
            set => SetProperty(ref _currentRoi, value);
        }

        private bool _isDrawingRoi;
        public bool IsDrawingRoi
        {
            get => _isDrawingRoi;
            set => SetProperty(ref _isDrawingRoi, value);
        }

        private Point _startPoint;

        public ICommand MouseDownCommand { get; }
        public ICommand MouseMoveCommand { get; }
        public ICommand MouseUpCommand { get; }

        public RoiViewModel()
        {
            MouseDownCommand = new RelayCommand(ExecuteMouseDown);
            MouseMoveCommand = new RelayCommand(ExecuteMouseMove);
            MouseUpCommand = new RelayCommand(ExecuteMouseUp);
        }

        private void ExecuteMouseDown(object? parameter)
        {
            if (parameter is MouseButtonEventArgs e)
            {
                _startPoint = e.GetPosition(e.Source as IInputElement);
                IsDrawingRoi = true;
                (e.Source as UIElement)?.CaptureMouse();
            }
        }

        private void ExecuteMouseMove(object? parameter)
        {
            if (IsDrawingRoi && parameter is MouseEventArgs e)
            {
                Point currentPoint = e.GetPosition(e.Source as IInputElement);

                double x = Math.Min(_startPoint.X, currentPoint.X);
                double y = Math.Min(_startPoint.Y, currentPoint.Y);
                double width = Math.Abs(_startPoint.X - currentPoint.X);
                double height = Math.Abs(_startPoint.Y - currentPoint.Y);

                CurrentRoi = new RoiModel(x, y, width, height);
            }
        }

        private void ExecuteMouseUp(object? parameter)
        {
            if (IsDrawingRoi)
            {
                IsDrawingRoi = false;
                (parameter as UIElement)?.ReleaseMouseCapture();
            }
        }
    }
}