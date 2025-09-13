using ImaGy.ViewModels;
using System;
using System.Windows;

namespace ImaGy.View
{
    public partial class MinimapWindow : Window
    {
        private MinimapViewModel? viewModel => DataContext as MinimapViewModel;

        public MinimapWindow()
        {
            InitializeComponent();
        }

        private void MinimapCanvas_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (viewModel != null)
            {
                viewModel.MinimapActualWidth = e.NewSize.Width;
                viewModel.MinimapActualHeight = e.NewSize.Height;
            }
        }

        private void MinimapWindow_Closed(object? sender, EventArgs e)
        { 
            viewModel?.Cleanup();
        }
    }
}
