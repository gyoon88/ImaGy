namespace ImaGy.Grids;

/// <summary>
/// 2D scalar grid stored row-major: index = row * Cols + col.
/// Coordinate convention (matches imshow / ROI JSON in this project):
/// column index = x (horizontal, left to right), row index = y (vertical, top to bottom; row 0 is top).
/// </summary>
public sealed class FloatGrid
{
    public int Rows { get; }
    public int Cols { get; }
    /// <summary>Row-major values; missing samples are double.NaN.</summary>
    public double[] Data { get; }
    /// <summary>Same length as Data; true where the source CSV had a finite value before alignment chain for this cell.</summary>
    public bool[] OriginallyFinite { get; }

    public FloatGrid(int rows, int cols, double[] data, bool[]? originallyFinite = null)
    {
        if (rows < 0 || cols < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (data.Length != rows * cols) throw new ArgumentException("data.Length must equal rows * cols.");
        Rows = rows;
        Cols = cols;
        Data = data;
        OriginallyFinite = originallyFinite ?? Enumerable.Range(0, data.Length).Select(i => IsFinite(data[i])).ToArray();
        if (OriginallyFinite.Length != data.Length) throw new ArgumentException("OriginallyFinite length mismatch.");
    }

    public static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    public int Length => Data.Length;

    public double this[int row, int col]
    {
        get => Data[row * Cols + col];
        set => Data[row * Cols + col] = value;
    }

    public FloatGrid Clone()
    {
        var d = new double[Data.Length];
        Array.Copy(Data, d, Data.Length);
        var o = new bool[OriginallyFinite.Length];
        Array.Copy(OriginallyFinite, o, OriginallyFinite.Length);
        return new FloatGrid(Rows, Cols, d, o);
    }

    public void CopyTo(FloatGrid target)
    {
        if (target.Rows != Rows || target.Cols != Cols) throw new ArgumentException("Shape mismatch.");
        Array.Copy(Data, target.Data, Data.Length);
        Array.Copy(OriginallyFinite, target.OriginallyFinite, OriginallyFinite.Length);
    }

    /// <summary>ScottPlot heatmap: first index row (y), second column (x).</summary>
    public double[,] ToDouble2D()
    {
        var z = new double[Rows, Cols];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                z[r, c] = this[r, c];
        return z;
    }
}
