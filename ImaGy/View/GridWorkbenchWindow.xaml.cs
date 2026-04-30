using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using ImaGy.Grids;
using ImaGy.ViewModels;

namespace ImaGy.View;

public partial class GridWorkbenchWindow : Window
{
    private const double MinDiffZoom = 0.25;
    private const double MaxDiffZoom = 24.0;

    private static bool IsShiftDown => (Keyboard.Modifiers & ModifierKeys.Shift) == ModifierKeys.Shift;
    private readonly List<(int C, int R)> _triangleCorners = new();
    private (int C, int R)? _rectAnchor;
    private (int C, int R)? _rectCurrent;
    private (int C, int R)? _circleCenter;
    private (int C, int R)? _circleEdge;
    private bool _draggingRect;
    private bool _draggingCircle;
    private bool _isDiffPanning;
    private System.Windows.Point _diffPanStartViewport;
    private double _diffPanStartHorizontalOffset;
    private double _diffPanStartVerticalOffset;
    private GridWorkbenchViewModel? _observedVm;

    public GridWorkbenchWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        HookVm(DataContext as GridWorkbenchViewModel);
        SyncCanvasToHost();
        RedrawRoiShapes();
    }

    protected override void OnClosed(EventArgs e)
    {
        HookVm(null);
        base.OnClosed(e);
    }

    private void HookVm(GridWorkbenchViewModel? vm)
    {
        if (_observedVm != null)
        {
            _observedVm.PropertyChanged -= VmOnPropertyChanged;
            _observedVm.DrawnRois.CollectionChanged -= VmOnDrawnRoisChanged;
        }

        _observedVm = vm;
        if (_observedVm != null)
        {
            _observedVm.PropertyChanged += VmOnPropertyChanged;
            _observedVm.DrawnRois.CollectionChanged += VmOnDrawnRoisChanged;
        }
    }

    private void VmOnPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(GridWorkbenchViewModel.SelectedRoi)
            or nameof(GridWorkbenchViewModel.PreviewDiff)
            or nameof(GridWorkbenchViewModel.RoiDrawTool)
            or nameof(GridWorkbenchViewModel.RoiJsonPath))
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                if (e.PropertyName == nameof(GridWorkbenchViewModel.PreviewDiff))
                    ResetDiffZoomToFit();
                SyncCanvasToHost();
                RedrawRoiShapes();
            }));
        }
    }

    private void VmOnDrawnRoisChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        Dispatcher.BeginInvoke(new Action(() =>
        {
            SyncCanvasToHost();
            RedrawRoiShapes();
        }));
    }

    private void DiffPreviewImage_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        SyncCanvasToHost();
        RedrawRoiShapes();
    }

    private void ResetDiffZoomToFit()
    {
        if (DiffPreviewImage.Source is not BitmapSource bmp || bmp.PixelWidth <= 0 || bmp.PixelHeight <= 0)
            return;
        if (DiffScrollViewer.ViewportWidth <= 0 || DiffScrollViewer.ViewportHeight <= 0)
            return;

        double fitX = DiffScrollViewer.ViewportWidth / bmp.PixelWidth;
        double fitY = DiffScrollViewer.ViewportHeight / bmp.PixelHeight;
        _diffZoom = Math.Clamp(Math.Min(fitX, fitY), MinDiffZoom, MaxDiffZoom);
        DiffZoomTransform.ScaleX = _diffZoom;
        DiffZoomTransform.ScaleY = _diffZoom;
        DiffScrollViewer.ScrollToHorizontalOffset(0);
        DiffScrollViewer.ScrollToVerticalOffset(0);
    }

    private double _scale = 1.0;
    private double _imgLeft;
    private double _imgTop;
    private int _gridCols;
    private int _gridRows;
    private double _diffZoom = 1.0;

    private void SyncCanvasToHost()
    {
        UpdateImageMetrics();
        double w = Math.Max(1, _gridCols * _diffZoom);
        double h = Math.Max(1, _gridRows * _diffZoom);
        RoiOverlayCanvas.Width = w;
        RoiOverlayCanvas.Height = h;
        DiffHostGrid.Width = w;
        DiffHostGrid.Height = h;
    }

    private void UpdateImageMetrics()
    {
        _scale = 0;
        _gridCols = 0;
        _gridRows = 0;
        if (DiffPreviewImage.Source is not BitmapSource bmp || bmp.PixelWidth <= 0)
            return;

        _gridCols = bmp.PixelWidth;
        _gridRows = bmp.PixelHeight;
        _scale = _diffZoom;
        _imgLeft = 0;
        _imgTop = 0;
    }

    private bool TryMapToCell(System.Windows.Point hostPos, out int col, out int row)
    {
        col = row = 0;
        UpdateImageMetrics();
        if (_scale <= 0) return false;

        double x = hostPos.X - _imgLeft;
        double y = hostPos.Y - _imgTop;
        double dispW = _gridCols * _scale;
        double dispH = _gridRows * _scale;
        if (x < 0 || y < 0 || x >= dispW || y >= dispH) return false;

        col = (int)Math.Clamp(Math.Floor(x / _scale), 0, _gridCols - 1);
        row = (int)Math.Clamp(Math.Floor(y / _scale), 0, _gridRows - 1);
        return true;
    }

    private void SetDiffZoom(double requestedZoom, System.Windows.Point viewportAnchor)
    {
        if (DiffPreviewImage.Source is not BitmapSource)
            return;

        double newZoom = Math.Clamp(requestedZoom, MinDiffZoom, MaxDiffZoom);
        if (Math.Abs(newZoom - _diffZoom) < 0.0001)
            return;

        // Keep the pixel under cursor fixed while zooming.
        double contentX = (DiffScrollViewer.HorizontalOffset + viewportAnchor.X) / _diffZoom;
        double contentY = (DiffScrollViewer.VerticalOffset + viewportAnchor.Y) / _diffZoom;

        _diffZoom = newZoom;
        DiffZoomTransform.ScaleX = _diffZoom;
        DiffZoomTransform.ScaleY = _diffZoom;
        SyncCanvasToHost();
        RedrawRoiShapes();
        UpdateLayout();

        DiffScrollViewer.ScrollToHorizontalOffset(Math.Max(0, contentX * _diffZoom - viewportAnchor.X));
        DiffScrollViewer.ScrollToVerticalOffset(Math.Max(0, contentY * _diffZoom - viewportAnchor.Y));
    }

    private void RedrawRoiShapes()
    {
        RoiOverlayCanvas.Children.Clear();
        UpdateImageMetrics();
        if (_scale <= 0 || DataContext is not GridWorkbenchViewModel vm) return;

        foreach (var roi in vm.DrawnRois)
        {
            bool selected = ReferenceEquals(roi, vm.SelectedRoi);
            DrawRoiDefinition(
                roi,
                selected ? System.Windows.Media.Color.FromRgb(50, 205, 50) : System.Windows.Media.Color.FromRgb(65, 105, 225),
                selected ? 2.0 : 1.25);
        }

        // 드래그 미리보기
        if (_draggingRect && _rectAnchor is { } a && _rectCurrent is { } b)
        {
            int xmin = Math.Min(a.C, b.C), xmax = Math.Max(a.C, b.C);
            int ymin = Math.Min(a.R, b.R), ymax = Math.Max(a.R, b.R);
            var prev = new GridRoiDefinition
            {
                Kind = RoiKind.Rectangle,
                X = xmin,
                Y = ymin,
                Width = xmax - xmin + 1,
                Height = ymax - ymin + 1
            };
            DrawRoiDefinition(prev, System.Windows.Media.Color.FromRgb(255, 215, 0), 1);
        }

        if (_draggingCircle && _circleCenter is { } cc && _circleEdge is { } ce)
        {
            double rad = Math.Sqrt((ce.C - cc.C) * (ce.C - cc.C) + (ce.R - cc.R) * (ce.R - cc.R));
            if (rad < 0.5) rad = 0.5;
            var prev = new GridRoiDefinition
            {
                Kind = RoiKind.Circle,
                CenterCol = cc.C,
                CenterRow = cc.R,
                RadiusPixels = rad
            };
            DrawRoiDefinition(prev, System.Windows.Media.Color.FromRgb(255, 215, 0), 1);
        }

        if (_triangleCorners.Count > 0)
        {
            var brush = new SolidColorBrush(System.Windows.Media.Color.FromRgb(255, 215, 0));
            for (int i = 0; i < _triangleCorners.Count; i++)
            {
                var ell = new Ellipse { Width = 6, Height = 6, Fill = brush, Stroke = System.Windows.Media.Brushes.White, StrokeThickness = 1 };
                var p = _triangleCorners[i];
                Canvas.SetLeft(ell, _imgLeft + p.C * _scale + _scale / 2 - 3);
                Canvas.SetTop(ell, _imgTop + p.R * _scale + _scale / 2 - 3);
                RoiOverlayCanvas.Children.Add(ell);
            }

            if (_triangleCorners.Count >= 2)
            {
                var line = new Line
                {
                    X1 = _imgLeft + _triangleCorners[0].C * _scale + _scale / 2,
                    Y1 = _imgTop + _triangleCorners[0].R * _scale + _scale / 2,
                    X2 = _imgLeft + _triangleCorners[1].C * _scale + _scale / 2,
                    Y2 = _imgTop + _triangleCorners[1].R * _scale + _scale / 2,
                    Stroke = brush,
                    StrokeThickness = 1
                };
                RoiOverlayCanvas.Children.Add(line);
            }

            if (_triangleCorners.Count >= 3)
            {
                var line2 = new Line
                {
                    X1 = _imgLeft + _triangleCorners[1].C * _scale + _scale / 2,
                    Y1 = _imgTop + _triangleCorners[1].R * _scale + _scale / 2,
                    X2 = _imgLeft + _triangleCorners[2].C * _scale + _scale / 2,
                    Y2 = _imgTop + _triangleCorners[2].R * _scale + _scale / 2,
                    Stroke = brush,
                    StrokeThickness = 1
                };
                RoiOverlayCanvas.Children.Add(line2);
            }
        }
    }

    private void DrawRoiDefinition(GridRoiDefinition roi, System.Windows.Media.Color stroke, double thick)
    {
        var brush = new SolidColorBrush(stroke);
        switch (roi.Kind)
        {
            case RoiKind.Rectangle:
                var rect = new System.Windows.Shapes.Rectangle
                {
                    Stroke = brush,
                    StrokeThickness = thick,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    Width = roi.Width * _scale,
                    Height = roi.Height * _scale
                };
                Canvas.SetLeft(rect, _imgLeft + roi.X * _scale);
                Canvas.SetTop(rect, _imgTop + roi.Y * _scale);
                RoiOverlayCanvas.Children.Add(rect);
                break;
            case RoiKind.Circle:
                double diam = 2 * roi.RadiusPixels * _scale;
                var ell = new Ellipse
                {
                    Stroke = brush,
                    StrokeThickness = thick,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    Width = diam,
                    Height = diam
                };
                Canvas.SetLeft(ell, _imgLeft + (roi.CenterCol - roi.RadiusPixels) * _scale);
                Canvas.SetTop(ell, _imgTop + (roi.CenterRow - roi.RadiusPixels) * _scale);
                RoiOverlayCanvas.Children.Add(ell);
                break;
            case RoiKind.Triangle:
                var poly = new Polygon { Stroke = brush, StrokeThickness = thick, Fill = System.Windows.Media.Brushes.Transparent};
                poly.Points.Add(new System.Windows.Point(_imgLeft + roi.C0 * _scale + _scale / 2, _imgTop + roi.R0 * _scale + _scale / 2));
                poly.Points.Add(new System.Windows.Point(_imgLeft + roi.C1 * _scale + _scale / 2, _imgTop + roi.R1 * _scale + _scale / 2));
                poly.Points.Add(new System.Windows.Point(_imgLeft + roi.C2 * _scale + _scale / 2, _imgTop + roi.R2 * _scale + _scale / 2));
                RoiOverlayCanvas.Children.Add(poly);
                break;
        }
    }

    private string CurrentTool => (DataContext as GridWorkbenchViewModel)?.RoiDrawTool ?? "없음";

    private void DiffScrollViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (DiffPreviewImage.Source is not BitmapSource)
            return;
        double factor = e.Delta > 0 ? 1.15 : 1 / 1.15;
        SetDiffZoom(_diffZoom * factor, e.GetPosition(DiffScrollViewer));
        e.Handled = true;
    }

    private void DiffScrollViewer_PreviewMouseDown(object sender, MouseButtonEventArgs e)
    {
        if (e.MiddleButton != MouseButtonState.Pressed)
            return;

        _isDiffPanning = true;
        _diffPanStartViewport = e.GetPosition(DiffScrollViewer);
        _diffPanStartHorizontalOffset = DiffScrollViewer.HorizontalOffset;
        _diffPanStartVerticalOffset = DiffScrollViewer.VerticalOffset;
        DiffScrollViewer.CaptureMouse();
        Cursor = System.Windows.Input.Cursors.SizeAll;
        e.Handled = true;
    }

    private void DiffScrollViewer_PreviewMouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (!_isDiffPanning)
            return;

        var current = e.GetPosition(DiffScrollViewer);
        var dx = current.X - _diffPanStartViewport.X;
        var dy = current.Y - _diffPanStartViewport.Y;
        DiffScrollViewer.ScrollToHorizontalOffset(Math.Max(0, _diffPanStartHorizontalOffset - dx));
        DiffScrollViewer.ScrollToVerticalOffset(Math.Max(0, _diffPanStartVerticalOffset - dy));
        e.Handled = true;
    }

    private void DiffScrollViewer_PreviewMouseUp(object sender, MouseButtonEventArgs e)
    {
        if (!_isDiffPanning || e.ChangedButton != MouseButton.Middle)
            return;

        _isDiffPanning = false;
        DiffScrollViewer.ReleaseMouseCapture();
        Cursor = System.Windows.Input.Cursors.Arrow;
        e.Handled = true;
    }

    private void RoiOverlay_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm || vm.PreviewDiff == null) return;
        if (!TryMapToCell(e.GetPosition(RoiOverlayCanvas), out int c, out int r))
        {
            vm.ResetDiffCursorInfo();
            return;
        }
        vm.UpdateDiffCursorInfo(c, r);
        if (!IsShiftDown)
            return;

        switch (CurrentTool)
        {
            case "사각형":
                _draggingRect = true;
                _rectAnchor = (c, r);
                _rectCurrent = (c, r);
                RoiOverlayCanvas.CaptureMouse();
                RedrawRoiShapes();
                break;
            case "원":
                if (!_draggingCircle)
                {
                    _draggingCircle = true;
                    _circleCenter = (c, r);
                    _circleEdge = (c, r);
                    RoiOverlayCanvas.CaptureMouse();
                }
                RedrawRoiShapes();
                break;
            case "삼각형":
                _triangleCorners.Add((c, r));
                if (_triangleCorners.Count >= 3)
                {
                    var p0 = _triangleCorners[0];
                    var p1 = _triangleCorners[1];
                    var p2 = _triangleCorners[2];
                    vm.AddDrawnRoi(new GridRoiDefinition
                    {
                        Kind = RoiKind.Triangle,
                        C0 = p0.C, R0 = p0.R,
                        C1 = p1.C, R1 = p1.R,
                        C2 = p2.C, R2 = p2.R
                    });
                    _triangleCorners.Clear();
                }
                RedrawRoiShapes();
                break;
        }
    }

    private void RoiOverlay_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm || vm.PreviewDiff == null) return;
        if (!TryMapToCell(e.GetPosition(RoiOverlayCanvas), out int c, out int r))
        {
            vm.ResetDiffCursorInfo();
            return;
        }
        vm.UpdateDiffCursorInfo(c, r);

        if (_draggingRect && _rectAnchor != null)
        {
            _rectCurrent = (c, r);
            RedrawRoiShapes();
        }
        else if (_draggingCircle && _circleCenter != null)
        {
            _circleEdge = (c, r);
            RedrawRoiShapes();
        }
    }

    private void RoiOverlay_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm) return;

        bool wasRect = _draggingRect;
        bool wasCircle = _draggingCircle;
        var ra = _rectAnchor;
        var rb = _rectCurrent;
        var cc = _circleCenter;
        var ce = _circleEdge;

        _draggingRect = false;
        _draggingCircle = false;
        _rectAnchor = _rectCurrent = null;
        _circleCenter = _circleEdge = null;
        RoiOverlayCanvas.ReleaseMouseCapture();

        if (wasRect && ra is { } a && rb is { } b)
        {
            int xmin = Math.Min(a.C, b.C), xmax = Math.Max(a.C, b.C);
            int ymin = Math.Min(a.R, b.R), ymax = Math.Max(a.R, b.R);
            vm.AddDrawnRoi(new GridRoiDefinition
            {
                Kind = RoiKind.Rectangle,
                X = xmin,
                Y = ymin,
                Width = xmax - xmin + 1,
                Height = ymax - ymin + 1
            });
        }
        else if (wasCircle && cc is { } cCenter && ce is { } cEdge)
        {
            double rad = Math.Sqrt((cEdge.C - cCenter.C) * (cEdge.C - cCenter.C) + (cEdge.R - cCenter.R) * (cEdge.R - cCenter.R));
            if (rad < 0.5) rad = 0.5;
            vm.AddDrawnRoi(new GridRoiDefinition
            {
                Kind = RoiKind.Circle,
                CenterCol = cCenter.C,
                CenterRow = cCenter.R,
                RadiusPixels = rad
            });
        }

        RedrawRoiShapes();
    }

    private void RoiOverlay_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        _triangleCorners.Clear();
        _draggingRect = false;
        _draggingCircle = false;
        _rectAnchor = _rectCurrent = null;
        _circleCenter = _circleEdge = null;
        RoiOverlayCanvas.ReleaseMouseCapture();
        RedrawRoiShapes();
    }

    private void RoiOverlay_MouseLeave(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (DataContext is GridWorkbenchViewModel vm)
            vm.ResetDiffCursorInfo();
    }

    private void ClearRoiOverlay_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is GridWorkbenchViewModel vm)
            vm.ClearDrawnRois();
        _triangleCorners.Clear();
        _draggingRect = false;
        _draggingCircle = false;
        RoiOverlayCanvas.ReleaseMouseCapture();
        RedrawRoiShapes();
    }

    private void SaveRoiJson_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm) return;
        try
        {
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string initialDir = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            string fileName = "roi_catalog_" + stamp + ".json";
            if (!string.IsNullOrWhiteSpace(vm.RoiJsonPath))
            {
                try
                {
                    var d = System.IO.Path.GetDirectoryName(vm.RoiJsonPath);
                    if (!string.IsNullOrEmpty(d) && Directory.Exists(d))
                        initialDir = d;
                    var baseName = System.IO.Path.GetFileNameWithoutExtension(vm.RoiJsonPath);
                    if (!string.IsNullOrEmpty(baseName))
                        fileName = baseName + "_" + stamp + ".json";
                }
                catch
                {
                    // keep defaults
                }
            }

            var dlg = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "ROI 카탈로그 JSON|*.json|All|*.*",
                Title = "ROI JSON 다른 이름으로 저장",
                InitialDirectory = initialDir,
                FileName = fileName,
                DefaultExt = ".json",
                AddExtension = true,
                OverwritePrompt = true
            };
            if (dlg.ShowDialog() != true)
                return;
            vm.SaveDrawnRoiToJson(dlg.FileName);
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show(ex.Message, "ROI 저장", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OpenLineProfile_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm) return;
        if (vm.OpenLineProfileCommand.CanExecute(null))
            vm.OpenLineProfileCommand.Execute(null);
    }

    private void OpenRoiStatsWindow_Click(object sender, RoutedEventArgs e)
    {
        var w = new GridRoiStatsWindow
        {
            Owner = this,
            DataContext = DataContext
        };
        w.Show();
    }

    private void OpenBatchWindow_Click(object sender, RoutedEventArgs e)
    {
        var w = new GridBatchWindow
        {
            Owner = this,
            DataContext = DataContext
        };
        w.Show();
    }

    private void OpenHelpWindow_Click(object sender, RoutedEventArgs e)
    {
        var w = new GridHelpWindow
        {
            Owner = this,
            DataContext = DataContext
        };
        w.Show();
    }

    private void AddOrUpdateRoi_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm) return;
        if (_draggingRect || _draggingCircle || _triangleCorners.Count > 0)
        {
            System.Windows.MessageBox.Show("ROI 그리기 동작을 완료한 뒤 추가/갱신 하세요.", "ROI", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        if (vm.SelectedRoi == null)
        {
            System.Windows.MessageBox.Show("먼저 Diff 영역에서 ROI를 그리세요.", "ROI", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        vm.UpsertCurrentRoi(vm.SelectedRoi);
        RedrawRoiShapes();
    }

    private void RemoveSelectedRoi_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm) return;
        vm.RemoveSelectedRoi();
        RedrawRoiShapes();
    }

    private void OpenRoiHypothesis_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not GridWorkbenchViewModel vm) return;
        if (vm.CurrentPipelineResult == null || vm.DrawnRois.Count < 2)
        {
            System.Windows.MessageBox.Show("Diff 격자를 먼저 표시(파이프라인 또는 Diff CSV)하고, ROI를 최소 2개 그려 주세요.",
                "Diff ROI 2그룹 검정", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
            return;
        }
        GridRoiHypothesisWindow.ShowForWorkbench(this, vm);
    }
}
