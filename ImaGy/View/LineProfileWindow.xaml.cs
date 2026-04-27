using ImaGy.Grids;
using ImaGy.ViewModels;
using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ImaGy.View;

public partial class LineProfileWindow : Window
{
    private LineProfileViewModel? _viewModel;

    public LineProfileWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        _viewModel = DataContext as LineProfileViewModel;
        if (_viewModel == null) return;
        _viewModel.PropertyChanged += ViewModelOnPropertyChanged;
        DrawProfile();
    }

    private void ViewModelOnPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(LineProfileViewModel.ProfileValues)
            or nameof(LineProfileViewModel.ValueMin)
            or nameof(LineProfileViewModel.ValueMax)
            or nameof(LineProfileViewModel.AxisMode))
        {
            DrawProfile();
        }
    }

    private void ProfileCanvas_SizeChanged(object sender, SizeChangedEventArgs e) => DrawProfile();

    private void DrawProfile()
    {
        ProfileCanvas.Children.Clear();
        if (_viewModel == null) return;
        var data = _viewModel.ProfileValues;
        if (data.Length < 1 || ProfileCanvas.ActualWidth <= 2 || ProfileCanvas.ActualHeight <= 2) return;

        double w = ProfileCanvas.ActualWidth;
        double h = ProfileCanvas.ActualHeight;
        double min = _viewModel.ValueMin;
        double max = _viewModel.ValueMax;
        double span = Math.Max(1e-12, max - min);

        var yAxis = new Line { X1 = 0, Y1 = 0, X2 = 0, Y2 = h, Stroke = System.Windows.Media.Brushes.WhiteSmoke, StrokeThickness = 1 };
        var xAxis = new Line { X1 = 0, Y1 = h, X2 = w, Y2 = h, Stroke = System.Windows.Media.Brushes.WhiteSmoke, StrokeThickness = 1 };
        ProfileCanvas.Children.Add(yAxis);
        ProfileCanvas.Children.Add(xAxis);

        Polyline? current = null;
        int denomX = Math.Max(1, data.Length - 1);
        for (int i = 0; i < data.Length; i++)
        {
            double v = data[i];
            if (!FloatGrid.IsFinite(v))
            {
                current = null;
                continue;
            }

            double x = (i / (double)denomX) * (w - 1);
            double y = h - ((v - min) / span) * h;
            y = Math.Clamp(y, 0, h);

            if (current == null)
            {
                current = new Polyline
                {
                    Stroke = System.Windows.Media.Brushes.LimeGreen,
                    StrokeThickness = 1.5,
                    SnapsToDevicePixels = true
                };
                ProfileCanvas.Children.Add(current);
            }
            current.Points.Add(new System.Windows.Point(x, y));
        }
    }

    private void ProfileCanvas_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (_viewModel == null || _viewModel.ProfileValues.Length == 0 || ProfileCanvas.ActualWidth <= 0)
            return;

        double x = e.GetPosition(ProfileCanvas).X;
        int n = _viewModel.ProfileValues.Length;
        if (n <= 0) return;
        int denomX = Math.Max(1, n - 1);
        int idx = (int)Math.Round((x / Math.Max(1, ProfileCanvas.ActualWidth)) * denomX);
        idx = Math.Clamp(idx, 0, n - 1);
        double v = _viewModel.ProfileValues[idx];
        string vText = FloatGrid.IsFinite(v) ? v.ToString("F4") : "합 없음(마스크/유효 셀 없음)";
        ProfileHoverText.Text = $"{_viewModel.AbscissaName}={idx}, 누적 합={vText}";
    }

    private void ProfileCanvas_MouseLeave(object sender, System.Windows.Input.MouseEventArgs e)
    {
        ProfileHoverText.Text = "마우스를 올려 라인 값 확인";
    }
}
