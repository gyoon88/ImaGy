using ImaGy.Grids;

namespace ImaGy.ViewModels;

public sealed class GridHistogramViewModel : BaseViewModel
{
    public int[] Bins { get; }
    public int MaxBin { get; }
    /// <summary>ProgressBar Maximum must be &gt; 0.</summary>
    public int MaxBinForUi => Math.Max(1, MaxBin);
    public GridStatisticsResult Stats { get; }
    public int BinCount => Bins.Length;

    public GridHistogramViewModel(FloatGrid grid, bool[]? mask, double? valueMin, double? valueMax, int binCount = 128)
    {
        Bins = GridStatisticsService.ComputeHistogram(grid, mask, valueMin, valueMax, binCount);
        MaxBin = Bins.Length == 0 ? 0 : Bins.Max();
        Stats = GridStatisticsService.Compute(grid, mask, valueMin, valueMax);
    }
}
