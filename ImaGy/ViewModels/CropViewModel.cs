using ImaGy.Models;
using ImaGy.Services;
using System.Windows.Media.Imaging;
using System.Windows.Input;
using System.Windows;
using ImaGy.ViewModels.Commands;

namespace ImaGy.ViewModels
{
    public class CropViewModel : BaseViewModel
    {
        private readonly MainViewModel _mainViewModel;
        private readonly CropService _cropService;

        public CropViewModel(MainViewModel mainViewModel, CropService cropService)
        {
            _mainViewModel = mainViewModel;
            _cropService = cropService;
            ApplyCropCommand = new RelayCommand<object>(ExecuteApplyCrop); // Added
        }

        public ICommand ApplyCropCommand { get; }

        private void ExecuteApplyCrop(object? parameter)
        {
            if (_mainViewModel.AfterImage != null && _mainViewModel.RoiViewModel.CurrentRoi != null)
            {
                var croppedImage = _cropService.CropImage(_mainViewModel.AfterImage, _mainViewModel.RoiViewModel.CurrentRoi);
                if (croppedImage != null)
                {
                    _mainViewModel.AfterImage = croppedImage;
                    _mainViewModel.BeforeImage = croppedImage; // Also update before image for consistency
                    _mainViewModel.RoiViewModel.CurrentRoi = null; // Clear ROI after cropping
                    _mainViewModel.RoiViewModel.IsDrawingRoi = false; // Exit ROI drawing mode
                }
            }
        }
    }
}