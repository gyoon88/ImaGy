using ImaGy.Grids;
using System;
using System.Linq;

namespace ImaGy.ViewModels;

public sealed class LineProfileViewModel : BaseViewModel
{
    private readonly FloatGrid _grid;
    private readonly bool[]? _mask;
    private string _axisMode = "Row";
    private int _index;
    private int _maxIndex;
    private double[] _profileValues = Array.Empty<double>();
    private double _valueMin;
    private double _valueMax = 1;
    private double _mean;

    public LineProfileViewModel(FloatGrid grid, bool[]? mask)
    {
        _grid = grid;
        _mask = mask;
        _maxIndex = Math.Max(0, grid.Rows - 1);
        _index = Math.Clamp(grid.Rows / 2, 0, _maxIndex);
        UpdateProfile();
    }

    public string AxisMode
    {
        get => _axisMode;
        set
        {
            if (!SetProperty(ref _axisMode, value))
                return;

            MaxIndex = _axisMode == "Column"
                ? Math.Max(0, _grid.Cols - 1)
                : Math.Max(0, _grid.Rows - 1);
            Index = Math.Clamp(Index, 0, MaxIndex);
            OnPropertyChanged(nameof(AxisLabel));
            OnPropertyChanged(nameof(IndexLabel));
            UpdateProfile();
        }
    }

    public int Index
    {
        get => _index;
        set
        {
            int clamped = Math.Clamp(value, 0, MaxIndex);
            if (!SetProperty(ref _index, clamped))
                return;
            UpdateProfile();
        }
    }

    public int MaxIndex
    {
        get => _maxIndex;
        private set => SetProperty(ref _maxIndex, value);
    }

    public string AxisLabel => AxisMode == "Column" ? "row" : "col";
    public string IndexLabel => AxisMode == "Column" ? "열 인덱스" : "행 인덱스";

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
            ? BuildColumn(Index)
            : BuildRow(Index);

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

    private double[] BuildRow(int row)
    {
        var vals = new double[_grid.Cols];
        for (int c = 0; c < _grid.Cols; c++)
        {
            int idx = row * _grid.Cols + c;
            bool maskedOut = _mask != null && (idx >= _mask.Length || !_mask[idx]);
            vals[c] = maskedOut ? double.NaN : _grid[row, c];
        }
        return vals;
    }

    private double[] BuildColumn(int col)
    {
        var vals = new double[_grid.Rows];
        for (int r = 0; r < _grid.Rows; r++)
        {
            int idx = r * _grid.Cols + col;
            bool maskedOut = _mask != null && (idx >= _mask.Length || !_mask[idx]);
            vals[r] = maskedOut ? double.NaN : _grid[r, col];
        }
        return vals;
    }
}
