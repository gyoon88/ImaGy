namespace ImaGy.Grids;

public static class GridPreprocessService
{
    public static void Apply(FloatGrid grid, GridPreprocessOptions opt)
    {
        if (opt.Order == PreprocessOrder.FillThenFilter)
        {
            ApplyFill(grid, opt);
            ApplyOutliers(grid, opt);
        }
        else
        {
            ApplyOutliers(grid, opt);
            ApplyFill(grid, opt);
        }
    }

    private static void ApplyOutliers(FloatGrid grid, GridPreprocessOptions opt)
    {
        switch (opt.OutlierMethod)
        {
            case OutlierMethod.None:
                return;
            case OutlierMethod.IqrTukey:
                ApplyIqr(grid, opt.IqrK);
                break;
            case OutlierMethod.NSigma:
                ApplyNSigma(grid, opt.SigmaMultiplier);
                break;
        }
    }

    private static void ApplyIqr(FloatGrid grid, double k)
    {
        var vals = grid.Data.Where(FloatGrid.IsFinite).OrderBy(x => x).ToArray();
        if (vals.Length < 4) return;
        int n = vals.Length;
        double q1 = PercentileSorted(vals, 0.25);
        double q3 = PercentileSorted(vals, 0.75);
        double iqr = q3 - q1;
        double lo = q1 - k * iqr;
        double hi = q3 + k * iqr;
        for (int i = 0; i < grid.Length; i++)
        {
            if (!FloatGrid.IsFinite(grid.Data[i])) continue;
            if (grid.Data[i] < lo || grid.Data[i] > hi)
                grid.Data[i] = double.NaN;
        }
    }

    private static double PercentileSorted(double[] sorted, double p)
    {
        if (sorted.Length == 0) return double.NaN;
        double pos = (sorted.Length - 1) * p;
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    private static void ApplyNSigma(FloatGrid grid, double nSigma)
    {
        var finite = grid.Data.Where(FloatGrid.IsFinite).ToArray();
        if (finite.Length == 0) return;
        double mean = finite.Average();
        double var = finite.Sum(x => (x - mean) * (x - mean)) / finite.Length;
        double std = Math.Sqrt(var);
        if (std < 1e-30) return;
        double lo = mean - nSigma * std;
        double hi = mean + nSigma * std;
        for (int i = 0; i < grid.Length; i++)
        {
            if (!FloatGrid.IsFinite(grid.Data[i])) continue;
            if (grid.Data[i] < lo || grid.Data[i] > hi)
                grid.Data[i] = double.NaN;
        }
    }

    private static void ApplyFill(FloatGrid grid, GridPreprocessOptions opt)
    {
        int w = Math.Max(1, opt.ConvolutionWindowSize);
        if (w % 2 == 0) w++;
        for (int iter = 0; iter < opt.FillIterations; iter++)
        {
            switch (opt.FillMethod)
            {
                case FillMethod.None:
                    return;
                case FillMethod.ConvolutionMean:
                    FillConvolutionMean(grid, w);
                    break;
                case FillMethod.LinearInterpolate1D:
                    FillLinear1D(grid);
                    break;
            }
        }
    }

    private static void FillConvolutionMean(FloatGrid grid, int window)
    {
        int r = window / 2;
        var next = (double[])grid.Data.Clone();
        for (int row = 0; row < grid.Rows; row++)
        {
            for (int col = 0; col < grid.Cols; col++)
            {
                int i = row * grid.Cols + col;
                if (FloatGrid.IsFinite(grid.Data[i])) { next[i] = grid.Data[i]; continue; }
                double sum = 0;
                int cnt = 0;
                for (int dr = -r; dr <= r; dr++)
                {
                    for (int dc = -r; dc <= r; dc++)
                    {
                        int rr = row + dr, cc = col + dc;
                        if (rr < 0 || rr >= grid.Rows || cc < 0 || cc >= grid.Cols) continue;
                        double v = grid.Data[rr * grid.Cols + cc];
                        if (FloatGrid.IsFinite(v)) { sum += v; cnt++; }
                    }
                }
                next[i] = cnt > 0 ? sum / cnt : double.NaN;
            }
        }
        Array.Copy(next, grid.Data, next.Length);
    }

    private static void FillLinear1D(FloatGrid grid)
    {
        var work = (double[])grid.Data.Clone();
        for (int row = 0; row < grid.Rows; row++)
            InterpolateRowFillNaN(work, grid, row);
        var afterRow = (double[])work.Clone();
        for (int col = 0; col < grid.Cols; col++)
            InterpolateColFillNaN(work, afterRow, grid, col);
        for (int i = 0; i < grid.Length; i++)
        {
            if (FloatGrid.IsFinite(grid.Data[i])) continue;
            if (FloatGrid.IsFinite(work[i])) grid.Data[i] = work[i];
        }
    }

    private static void InterpolateRowFillNaN(double[] work, FloatGrid grid, int row)
    {
        int cols = grid.Cols;
        var xs = new List<int>();
        var ys = new List<double>();
        for (int c = 0; c < cols; c++)
        {
            double v = grid.Data[row * cols + c];
            if (FloatGrid.IsFinite(v)) { xs.Add(c); ys.Add(v); }
        }
        if (xs.Count == 0) return;
        for (int c = 0; c < cols; c++)
        {
            int i = row * cols + c;
            if (FloatGrid.IsFinite(grid.Data[i])) continue;
            work[i] = xs.Count == 1 ? ys[0] : LinearInterp(c, xs, ys);
        }
    }

    private static void InterpolateColFillNaN(double[] work, double[] afterRow, FloatGrid grid, int col)
    {
        int cols = grid.Cols;
        var xs = new List<int>();
        var ys = new List<double>();
        for (int r = 0; r < grid.Rows; r++)
        {
            double v = afterRow[r * cols + col];
            if (FloatGrid.IsFinite(v)) { xs.Add(r); ys.Add(v); }
        }
        if (xs.Count == 0) return;
        for (int r = 0; r < grid.Rows; r++)
        {
            int i = r * cols + col;
            if (FloatGrid.IsFinite(grid.Data[i])) continue;
            double iy = xs.Count == 1 ? ys[0] : LinearInterp(r, xs, ys);
            if (!FloatGrid.IsFinite(afterRow[i])) work[i] = iy;
            else work[i] = (afterRow[i] + iy) * 0.5;
        }
    }

    private static double LinearInterp(int x, List<int> knots, List<double> vals)
    {
        if (knots.Count == 0) return double.NaN;
        if (x <= knots[0]) return vals[0];
        if (x >= knots[^1]) return vals[^1];
        int k = 0;
        while (k < knots.Count - 1 && !(knots[k] <= x && x <= knots[k + 1])) k++;
        if (k >= knots.Count - 1) return vals[^1];
        int x0 = knots[k], x1 = knots[k + 1];
        double y0 = vals[k], y1 = vals[k + 1];
        if (x1 == x0) return y0;
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
}
