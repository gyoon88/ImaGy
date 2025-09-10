using System.Windows;
using System.Windows.Controls;
using ImaGy.ViewModels;
using ImaGy.Services;
using System.Windows.Media.Imaging;
using System.Windows.Input;

namespace ImaGy.View
{
    public partial class MainWindow : Window
    {
        private bool _isSyncingScroll = false;

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            if (this.DataContext is MainViewModel viewModel)
            {
                // ViewModel의 서비스가 View의 ScrollViewer를 제어할 수 있도록 Action을 등록
                viewModel.ImageDisplay.RequestScrollAction = (h, v) =>
                {
                    _isSyncingScroll = true;
                    BeforeImageScrollViewer.ScrollToHorizontalOffset(h);
                    BeforeImageScrollViewer.ScrollToVerticalOffset(v);
                    AfterImageScrollViewer.ScrollToHorizontalOffset(h);
                    AfterImageScrollViewer.ScrollToVerticalOffset(v);
                    _isSyncingScroll = false;
                };

                // 이미지 로드 시 화면에 맞추는 로직
                viewModel.ImageLoadedCallback = (BitmapSource img) =>
                {
                    // 뷰가 완전히 로드된 후에 크기를 계산하도록 Dispatcher 사용
                    Dispatcher.InvokeAsync(() => {
                        viewModel.ImageDisplay.ResetDisplay(
                            img.PixelWidth,
                            img.PixelHeight,
                            BeforeImageScrollViewer.ActualWidth,
                            BeforeImageScrollViewer.ActualHeight
                        );
                    });
                };
            }
        }

        // 두 ScrollViewer의 위치를 동기화하고 ViewModel에 현재 위치를 알려주는 이벤트
        private void ScrollViewer_Sync(object sender, ScrollChangedEventArgs e)
        {
            if (_isSyncingScroll || this.DataContext is not MainViewModel viewModel) return;
            
            _isSyncingScroll = true;

            var scrollViewerToSync = sender == BeforeImageScrollViewer ? AfterImageScrollViewer : BeforeImageScrollViewer;
            scrollViewerToSync.ScrollToHorizontalOffset(e.HorizontalOffset);
            scrollViewerToSync.ScrollToVerticalOffset(e.VerticalOffset);

            // ViewModel의 상태 업데이트 (미니맵 등을 위해)
            viewModel.ScrollViewerHorizontalOffset = e.HorizontalOffset;
            viewModel.ScrollViewerVerticalOffset = e.VerticalOffset;
            viewModel.ScrollViewerViewportWidth = e.ViewportWidth;
            viewModel.ScrollViewerViewportHeight = e.ViewportHeight;

            _isSyncingScroll = false;
        }
    }
}