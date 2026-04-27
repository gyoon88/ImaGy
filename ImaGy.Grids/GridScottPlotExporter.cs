using ScottPlot;

namespace ImaGy.Grids;

public static class GridScottPlotExporter
{
    public static void SaveMatplotlibStyleHeatmap(
        string path,
        double[,] z,
        double vmin,
        double vmax,
        string title,
        string colormapId,
        int widthPx,
        int heightPx)
    {
        var plt = new Plot();
        var hm = plt.Add.Heatmap(z);
        hm.Colormap = MapColormap(colormapId);
        hm.ManualRange = new(vmin, vmax);
        plt.Title(title);
        plt.Axes.Left.Label.Text = "row (y)";
        plt.Axes.Bottom.Label.Text = "column (x)";
        plt.Add.ColorBar(hm);
        plt.SavePng(path, widthPx, heightPx);
    }

    public static void SaveHistogramPng(string path, int[] counts, string title)
    {
        var plt = new Plot();
        double[] xs = Enumerable.Range(0, counts.Length).Select(i => (double)i).ToArray();
        double[] ys = counts.Select(c => (double)c).ToArray();
        plt.Add.Bars(xs, ys);
        plt.Title(title);
        plt.YLabel("count");
        plt.XLabel("bin");
        plt.SavePng(path, 800, 500);
    }

    private static IColormap MapColormap(string id) =>
        id.Trim().ToUpperInvariant() switch
        {
            "VIRIDIS" => new ScottPlot.Colormaps.Viridis(),
            "JET" => new ScottPlot.Colormaps.Jet(),
            "GRAY" => new ScottPlot.Colormaps.Custom(
                [ScottPlot.Color.FromHex("#000000"), ScottPlot.Color.FromHex("#FFFFFF")], false),
            "HOT" or "COOL" => new ScottPlot.Colormaps.Turbo(),
            _ => new ScottPlot.Colormaps.Turbo()
        };
}
