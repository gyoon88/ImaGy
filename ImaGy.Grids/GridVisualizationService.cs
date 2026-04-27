using OpenCvSharp;

namespace ImaGy.Grids;

public sealed class GridVisualizationOptions
{
    public double? ClipMin { get; set; }
    public double? ClipMax { get; set; }
    /// <summary>If true, heatmap hides (alpha=0) pixels outside clip range instead of saturating.</summary>
    public bool MaskOutsideClipForHeatmap { get; set; } = true;
    public double PercentileLow { get; set; } = 2;
    public double PercentileHigh { get; set; } = 98;
}

public static class GridVisualizationService
{
    public static (double lo, double hi) GetNormalizeRange(FloatGrid grid, GridVisualizationOptions opt)
    {
        var f = grid.Data.Where(FloatGrid.IsFinite).ToArray();
        if (f.Length == 0) return (0, 1);
        Array.Sort(f);
        double lo = Percentile(f, opt.PercentileLow);
        double hi = Percentile(f, opt.PercentileHigh);
        if (hi <= lo) hi = lo + 1e-9;
        return (lo, hi);
    }

    private static double Percentile(double[] sorted, double p)
    {
        if (sorted.Length == 0) return double.NaN;
        double pos = (sorted.Length - 1) * (p / 100.0);
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        lo = Math.Clamp(lo, 0, sorted.Length - 1);
        hi = Math.Clamp(hi, 0, sorted.Length - 1);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    /// <summary>8-bit grayscale preview (row-major compatible with imshow orientation).</summary>
    public static Mat ToGray8Preview(FloatGrid grid, GridVisualizationOptions opt, double? vmin = null, double? vmax = null)
    {
        var (lo, hi) = (vmin, vmax) switch
        {
            (not null, not null) => (vmin!.Value, vmax!.Value),
            _ => GetNormalizeRange(grid, opt)
        };
        if (hi <= lo) hi = lo + 1e-9;
        var m = new Mat(grid.Rows, grid.Cols, MatType.CV_8UC1);
        for (int r = 0; r < grid.Rows; r++)
        {
            for (int c = 0; c < grid.Cols; c++)
            {
                double v = grid[r, c];
                byte b = 0;
                if (FloatGrid.IsFinite(v))
                {
                    double t = (v - lo) / (hi - lo);
                    t = Math.Clamp(t, 0, 1);
                    b = (byte)(t * 255);
                }
                m.Set(r, c, b);
            }
        }
        return m;
    }

    public static Mat RenderHeatmapBgra(
        FloatGrid grid,
        double vmin,
        double vmax,
        string colormapId,
        GridVisualizationOptions opt)
    {
        var cmap = GridColormapCatalog.Resolve(colormapId);
        bool intensityGray = GridColormapCatalog.IsIntensityOnly(colormapId);
        if (vmax <= vmin) vmax = vmin + 1e-9;
        using var gray = new Mat(grid.Rows, grid.Cols, MatType.CV_8UC1);
        using var maskIn = new Mat(grid.Rows, grid.Cols, MatType.CV_8UC1, Scalar.All(0));
        for (int r = 0; r < grid.Rows; r++)
        {
            for (int c = 0; c < grid.Cols; c++)
            {
                double v = grid[r, c];
                if (!FloatGrid.IsFinite(v)) continue;
                bool inRange = true;
                if (opt.ClipMin.HasValue && v < opt.ClipMin.Value) inRange = false;
                if (opt.ClipMax.HasValue && v > opt.ClipMax.Value) inRange = false;
                if (!inRange && opt.MaskOutsideClipForHeatmap)
                {
                    gray.Set(r, c, (byte)0);
                    continue;
                }
                double vMap = v;
                if (!inRange && !opt.MaskOutsideClipForHeatmap)
                {
                    if (opt.ClipMin.HasValue) vMap = Math.Max(vMap, opt.ClipMin.Value);
                    if (opt.ClipMax.HasValue) vMap = Math.Min(vMap, opt.ClipMax.Value);
                }
                double t = (vMap - vmin) / (vmax - vmin);
                t = Math.Clamp(t, 0, 1);
                gray.Set(r, c, (byte)(t * 255));
                maskIn.Set(r, c, (byte)255);
            }
        }

        using var color = new Mat();
        if (intensityGray)
            Cv2.CvtColor(gray, color, ColorConversionCodes.GRAY2BGR);
        else
            Cv2.ApplyColorMap(gray, color, cmap);
        var bgra = new Mat(grid.Rows, grid.Cols, MatType.CV_8UC4);
        Cv2.CvtColor(color, bgra, ColorConversionCodes.BGR2BGRA);
        for (int r = 0; r < grid.Rows; r++)
        {
            for (int c = 0; c < grid.Cols; c++)
            {
                var px = bgra.At<Vec4b>(r, c);
                byte a = maskIn.Get<byte>(r, c);
                if (a == 0) px[3] = 0;
                bgra.Set(r, c, px);
            }
        }
        return bgra;
    }

    public static void SaveHeatmapPng(FloatGrid grid, string path, double vmin, double vmax, string colormapId, GridVisualizationOptions opt, int? upscaleMaxDim = null)
    {
        using var bgra = RenderHeatmapBgra(grid, vmin, vmax, colormapId, opt);
        using var toSave = MaybeUpscale(bgra, upscaleMaxDim);
        Cv2.ImWrite(path, toSave);
    }

    private static Mat MaybeUpscale(Mat bgra, int? maxDim)
    {
        if (!maxDim.HasValue) return bgra.Clone();
        int m = Math.Max(bgra.Rows, bgra.Cols);
        if (m <= maxDim.Value) return bgra.Clone();
        double s = maxDim.Value / (double)m;
        var sz = new Size((int)(bgra.Cols * s), (int)(bgra.Rows * s));
        var dst = new Mat();
        Cv2.Resize(bgra, dst, sz, 0, 0, InterpolationFlags.Linear);
        return dst;
    }

    public static void SaveNormalizedRgbRaster(FloatGrid grid, string path, GridVisualizationOptions vis, double vmin, double vmax, int? upscaleMaxDim = null)
    {
        using var g8 = new Mat(grid.Rows, grid.Cols, MatType.CV_8UC1);
        if (vmax <= vmin) vmax = vmin + 1e-9;
        for (int r = 0; r < grid.Rows; r++)
        {
            for (int c = 0; c < grid.Cols; c++)
            {
                double v = grid[r, c];
                byte b = 0;
                if (FloatGrid.IsFinite(v))
                {
                    double t = (v - vmin) / (vmax - vmin);
                    b = (byte)(Math.Clamp(t, 0, 1) * 255);
                }
                g8.Set(r, c, b);
            }
        }
        using var bgr = new Mat();
        Cv2.CvtColor(g8, bgr, ColorConversionCodes.GRAY2BGR);
        using var toSave = MaybeUpscale(bgr, upscaleMaxDim);
        Cv2.ImWrite(path, toSave);
    }
}
