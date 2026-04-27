namespace ImaGy.Grids;

public static class GridCombineService
{
    public static FloatGrid Combine(FloatGrid a, FloatGrid b, GridCombineOptions opt)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException("Grids must have the same shape after alignment.");

        var data = new double[a.Length];
        var orig = new bool[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            bool bothNow = FloatGrid.IsFinite(a.Data[i]) && FloatGrid.IsFinite(b.Data[i]);
            bool origOverlap = a.OriginallyFinite[i] && b.OriginallyFinite[i];
            bool use = opt.MaskRule == ValidMaskRule.BothValid ? bothNow : origOverlap;
            if (!use)
            {
                data[i] = double.NaN;
                orig[i] = origOverlap;
                continue;
            }

            double va = a.Data[i];
            double vb = b.Data[i];
            double raw = opt.Operation == CombineOperation.Subtract ? va - vb : va + vb;
            double v = ApplyPresentation(raw, opt.Presentation);
            if (opt.ClipMin.HasValue || opt.ClipMax.HasValue)
            {
                double lo = opt.ClipMin ?? double.NegativeInfinity;
                double hi = opt.ClipMax ?? double.PositiveInfinity;
                if (FloatGrid.IsFinite(v))
                    v = Math.Clamp(v, lo, hi);
            }
            data[i] = v;
            orig[i] = origOverlap;
        }

        return new FloatGrid(a.Rows, a.Cols, data, orig);
    }

    private static double ApplyPresentation(double raw, DiffPresentation p) => p switch
    {
        DiffPresentation.Signed => raw,
        DiffPresentation.Abs => FloatGrid.IsFinite(raw) ? Math.Abs(raw) : double.NaN,
        DiffPresentation.PositiveOnly => FloatGrid.IsFinite(raw) && raw > 0 ? raw : (raw == 0 ? 0 : double.NaN),
        DiffPresentation.NegativeOnly => FloatGrid.IsFinite(raw) && raw < 0 ? raw : (raw == 0 ? 0 : double.NaN),
        _ => raw
    };

    /// <summary>Finite-range for heatmap (V-3): uses clip if set; else data percentiles; symmetric for signed if requested.</summary>
    public static (double vmin, double vmax) GetDisplayRange(FloatGrid diff, GridCombineOptions opt)
    {
        var finite = diff.Data.Where(FloatGrid.IsFinite).ToArray();
        if (finite.Length == 0) return (0, 1);

        if (opt.ClipMin.HasValue && opt.ClipMax.HasValue)
            return (opt.ClipMin.Value, opt.ClipMax.Value);

        double min = finite.Min();
        double max = finite.Max();
        if (opt.Presentation == DiffPresentation.Signed && opt.SymmetricVRangeForSigned)
        {
            double m = Math.Max(Math.Abs(min), Math.Abs(max));
            return (-m, m);
        }
        return (min, max);
    }
}
