using ImaGy.Grids;
using System;
using System.Linq;

namespace ImaGy.ViewModels;

/// <summary>
/// 1D 라인 **합** 프로파일: 히스토그램 "행/열 합"과 같이, 한 ROI 축을 따라 **다른 축에 대한 셀 값의 합**을 나열합니다.
/// "Row" 모드: 가로 축(그래프) = ROI y(행) 인덱스 0..H-1, 값 = 그 행에서 모든 x(열)에 대한 합(X 방향 누적).
/// "Column" 모드: 가로 축 = ROI x(열) 인덱스 0..W-1, 값 = 그 열에서 모든 y(행)에 대한 합(Y 방향 누적).
/// </summary>
public sealed class LineProfileViewModel : BaseViewModel
{
    private readonly FloatGrid _grid;
    private readonly bool[]? _mask;
    private string _axisMode = "Row";
    private double[] _profileValues = Array.Empty<double>();
    private double _valueMin;
    private double _valueMax = 1;
    private double _mean;

    public LineProfileViewModel(FloatGrid grid, bool[]? mask)
    {
        _grid = grid;
        _mask = mask;
        UpdateProfile();
    }

    public string AxisMode
    {
        get => _axisMode;
        set
        {
            if (!SetProperty(ref _axisMode, value))
                return;

            OnPropertyChanged(nameof(AxisAbscissaDescription));
            UpdateProfile();
        }
    }

    /// <summary>그래프 가로 축이 의미하는 것(ROI에서의 인덱스 축)</summary>
    public string AxisAbscissaDescription =>
        AxisMode == "Column"
            ? "가로 축: ROI x(열) 인덱스, 값: 해당 열에 대한 y(행) 방향 합"
            : "가로 축: ROI y(행) 인덱스, 값: 해당 행에 대한 x(열) 방향 합";

    /// <summary>호버 텍스트용. Column 모드면 ROI x(열) 인덱스, Row 모드면 ROI y(행) 인덱스.</summary>
    public string AbscissaName => AxisMode == "Column" ? "ROI x" : "ROI y";

    public double[] ProfileValues
    {
        get => _profileValues;
        private set => SetProperty(ref _profileValues, value);
    }

    public double ValueMin
    {
        get => _valueMin;
        private set => SetProperty(ref _valueMin, value);
    }

    public double ValueMax
    {
        get => _valueMax;
        private set => SetProperty(ref _valueMax, value);
    }

    public double Mean
    {
        get => _mean;
        private set => SetProperty(ref _mean, value);
    }

    private void UpdateProfile()
    {
        double[] values = AxisMode == "Column"
            ? BuildColumnSumsPerColIndex()
            : BuildRowSumsPerRowIndex();

        ProfileValues = values;

        var finite = values.Where(FloatGrid.IsFinite).ToArray();
        if (finite.Length == 0)
        {
            ValueMin = 0;
            ValueMax = 1;
            Mean = double.NaN;
            return;
        }

        ValueMin = finite.Min();
        ValueMax = finite.Max();
        if (Math.Abs(ValueMax - ValueMin) < 1e-12)
        {
            ValueMin -= 0.5;
            ValueMax += 0.5;
        }
        Mean = finite.Average();
    }

    private bool IsCellIncluded(int index)
    {
        if (_mask == null) return true;
        return (uint)index < (uint)_mask.Length && _mask[index];
    }

    /// <summary>각 <b>행(ROI y)</b> r에 대해, 그 행의 <b>모든 열</b> c에 대해 (마스크·유한) 값의 합 — 길이 = Rows. 그래프 가로 = r (y 인덱스).</summary>
    private double[] BuildRowSumsPerRowIndex()
    {
        var vals = new double[_grid.Rows];
        for (int r = 0; r < _grid.Rows; r++)
        {
            double s = 0;
            int n = 0;
            for (int c = 0; c < _grid.Cols; c++)
            {
                int idx = r * _grid.Cols + c;
                if (!IsCellIncluded(idx)) continue;
                var v = _grid[r, c];
                if (!FloatGrid.IsFinite(v)) continue;
                s += v;
                n++;
            }

            vals[r] = n == 0 ? double.NaN : s;
        }

        return vals;
    }

    /// <summary>각 <b>열(ROI x)</b> c에 대해, 그 열의 <b>모든 행</b> r에 대해 (마스크·유한) 값의 합 — 길이 = Cols. 그래프 가로 = c (x 인덱스).</summary>
    private double[] BuildColumnSumsPerColIndex()
    {
        var vals = new double[_grid.Cols];
        for (int c = 0; c < _grid.Cols; c++)
        {
            double s = 0;
            int n = 0;
            for (int r = 0; r < _grid.Rows; r++)
            {
                int idx = r * _grid.Cols + c;
                if (!IsCellIncluded(idx)) continue;
                var v = _grid[r, c];
                if (!FloatGrid.IsFinite(v)) continue;
                s += v;
                n++;
            }

            vals[c] = n == 0 ? double.NaN : s;
        }

        return vals;
    }
}
