namespace ImaGy.Grids;

public enum GridAlignmentMode
{
    Crop,
    Pad,
    Resample
}

public enum PadPlacement
{
    TopLeft,
    Center
}

public enum ResampleShapeReference
{
    GridA,
    GridB
}

public enum CombineOperation
{
    Subtract,
    Add
}

public enum DiffPresentation
{
    Signed,
    Abs,
    PositiveOnly,
    NegativeOnly
}

public enum ValidMaskRule
{
    /// <summary>After alignment and preprocess, both grids finite at cell.</summary>
    BothValid,
    /// <summary>Cells where original A and B were both finite (propagated through alignment).</summary>
    OriginalOverlap
}

public enum PreprocessOrder
{
    FillThenFilter,
    FilterThenFill
}

public enum OutlierMethod
{
    None,
    IqrTukey,
    NSigma
}

public enum FillMethod
{
    None,
    ConvolutionMean,
    LinearInterpolate1D
}
