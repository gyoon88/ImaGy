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
            viewModel.ImageLoadedCallback = (img) =>
            {
                if (img != null)
                {
                    viewModel.ImageDisplay.ResetDisplay(
                        img.PixelWidth,
                        img.PixelHeight,
                        BeforeImageScrollViewer.ActualWidth,
                        BeforeImageScrollViewer.ActualHeight
                    );
                }
            };

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
            if (isSyncing) return;

            isSyncing = true;

            var changedScrollViewer = (ScrollViewer)sender;
            var otherScrollViewer = changedScrollViewer == BeforeImageScrollViewer ? AfterImageScrollViewer : BeforeImageScrollViewer;
            otherScrollViewer.ScrollToHorizontalOffset(e.HorizontalOffset);
            otherScrollViewer.ScrollToVerticalOffset(e.VerticalOffset);

            // Update ViewModel for Minimap
            viewModel.ScrollViewerHorizontalOffset = e.HorizontalOffset;
            viewModel.ScrollViewerVerticalOffset = e.VerticalOffset;
            viewModel.ScrollViewerViewportWidth = e.ViewportWidth;
            viewModel.ScrollViewerViewportHeight = e.ViewportHeight;

            isSyncing = false;
        }
    }
}
