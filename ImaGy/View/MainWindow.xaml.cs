using System.Windows;
using System.Windows.Controls;
using ImaGy.ViewModels;

namespace ImaGy.View
{
    public partial class MainWindow : Window
    {
        private MainViewModel viewModel;
        private bool isSyncing; // Flag to prevent re-entrancy

        public MainWindow()
        {
            InitializeComponent();
            viewModel = (MainViewModel)DataContext;

            viewModel.ImageDisplay.RequestScrollAction = (h, v) =>
            {
                BeforeImageScrollViewer.ScrollToHorizontalOffset(h);
                BeforeImageScrollViewer.ScrollToVerticalOffset(v);
                AfterImageScrollViewer.ScrollToHorizontalOffset(h);
                AfterImageScrollViewer.ScrollToVerticalOffset(v);
            };
        }

        private void ScrollViewer_Sync(object sender, ScrollChangedEventArgs e)
        {
            // Final, robust scaling logic on the first ScrollChanged event after loading
            if (viewModel.IsImageLoading && viewModel.BeforeImage != null && e.ViewportWidth > 0)
            {
                viewModel.ImageDisplay.ResetDisplay(
                    viewModel.BeforeImage.PixelWidth,
                    viewModel.BeforeImage.PixelHeight,
                    e.ViewportWidth, // Use the final ViewportWidth
                    e.ViewportHeight // Use the final ViewportHeight
                );
                viewModel.IsImageLoading = false; // Finalize loading state
            }

            if (isSyncing) return;

            isSyncing = true;

            var changedScrollViewer = (ScrollViewer)sender;
            var otherScrollViewer = changedScrollViewer == BeforeImageScrollViewer ? AfterImageScrollViewer : BeforeImageScrollViewer;
            otherScrollViewer.ScrollToHorizontalOffset(e.HorizontalOffset);
            otherScrollViewer.ScrollToVerticalOffset(e.VerticalOffset);

            // Update ViewModel for Minimap
            viewModel.ImageDisplay.HorizontalOffset = e.HorizontalOffset;
            viewModel.ImageDisplay.VerticalOffset = e.VerticalOffset;
            viewModel.ImageDisplay.ViewportWidth = e.ViewportWidth;
            viewModel.ImageDisplay.ViewportHeight = e.ViewportHeight;

            isSyncing = false;
        }

        private void BeforeImageScrollViewer_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            // Initial, approximate scaling to trigger layout and potential scrollbars
            if (viewModel.IsImageLoading && viewModel.BeforeImage != null && e.NewSize.Width > 0)
            {
                viewModel.ImageDisplay.ResetDisplay(
                    viewModel.BeforeImage.PixelWidth,
                    viewModel.BeforeImage.PixelHeight,
                    e.NewSize.Width,
                    e.NewSize.Height
                );
            }
        }
    }
}
