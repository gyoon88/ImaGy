namespace ImaGy.Grids;

public sealed class GridPreprocessOptions
{
    public PreprocessOrder Order { get; set; } = PreprocessOrder.FillThenFilter;
    public FillMethod FillMethod { get; set; } = FillMethod.None;
    public int ConvolutionWindowSize { get; set; } = 3;
    public int FillIterations { get; set; } = 3;
    public OutlierMethod OutlierMethod { get; set; } = OutlierMethod.None;
    public double IqrK { get; set; } = 1.5;
    public double SigmaMultiplier { get; set; } = 3.0;
}
