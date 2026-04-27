namespace ImaGy.Grids;

public sealed class GridStatisticsResult
{
    public int Count { get; init; }
    public double Mean { get; init; }
    public double Std { get; init; }
    public double Min { get; init; }
    public double Max { get; init; }
    public double Median { get; init; }
    public double P25 { get; init; }
    public double P75 { get; init; }
}

public static class GridStatisticsService
{
    public static GridStatisticsResult Compute(FloatGrid grid, bool[]? mask, double? valueMin, double? valueMax)
    {
        var vals = new List<double>();
        for (int i = 0; i < grid.Length; i++)
        {
            if (mask != null && (i >= mask.Length || !mask[i])) continue;
            double v = grid.Data[i];
            if (!FloatGrid.IsFinite(v)) continue;
            if (valueMin.HasValue && v < valueMin.Value) continue;
            if (valueMax.HasValue && v > valueMax.Value) continue;
            vals.Add(v);
        }
        if (vals.Count == 0)
            return new GridStatisticsResult { Count = 0, Mean = double.NaN, Std = double.NaN, Min = double.NaN, Max = double.NaN, Median = double.NaN, P25 = double.NaN, P75 = double.NaN };

        vals.Sort();
        double mean = vals.Average();
        double var = vals.Sum(x => (x - mean) * (x - mean)) / vals.Count;
        return new GridStatisticsResult
        {
            Count = vals.Count,
            Mean = mean,
            Std = Math.Sqrt(var),
            Min = vals[0],
            Max = vals[^1],
            Median = PercentileSorted(vals, 0.5),
            P25 = PercentileSorted(vals, 0.25),
            P75 = PercentileSorted(vals, 0.75)
        };
    }

    private static double PercentileSorted(List<double> sorted, double p)
    {
        double pos = (sorted.Count - 1) * p;
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        lo = Math.Clamp(lo, 0, sorted.Count - 1);
        hi = Math.Clamp(hi, 0, sorted.Count - 1);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    public static int[] ComputeHistogram(FloatGrid grid, bool[]? mask, double? valueMin, double? valueMax, int binCount)
    {
        if (binCount < 1) binCount = 256;
        var vals = new List<double>();
        for (int i = 0; i < grid.Length; i++)
        {
            if (mask != null && (i >= mask.Length || !mask[i])) continue;
            double v = grid.Data[i];
            if (!FloatGrid.IsFinite(v)) continue;
            if (valueMin.HasValue && v < valueMin.Value) continue;
            if (valueMax.HasValue && v > valueMax.Value) continue;
            vals.Add(v);
        }
        var hist = new int[binCount];
        if (vals.Count == 0) return hist;
        vals.Sort();
        double lo = vals[0];
        double hi = vals[^1];
        if (hi <= lo) { hist[binCount / 2] = vals.Count; return hist; }
        foreach (var v in vals)
        {
            int b = (int)((v - lo) / (hi - lo) * (binCount - 1e-9));
            b = Math.Clamp(b, 0, binCount - 1);
            hist[b]++;
        }
        return hist;
    }
}
