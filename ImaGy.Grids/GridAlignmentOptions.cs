namespace ImaGy.Grids;

public sealed class GridAlignmentOptions
{
    public GridAlignmentMode Mode { get; set; } = GridAlignmentMode.Crop;
    public PadPlacement PadPlacement { get; set; } = PadPlacement.TopLeft;
    public ResampleShapeReference ResampleReference { get; set; } = ResampleShapeReference.GridA;
    public OpenCvSharp.InterpolationFlags Interpolation { get; set; } = OpenCvSharp.InterpolationFlags.Linear;
}
