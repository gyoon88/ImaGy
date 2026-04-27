namespace ImaGy.Grids;

public sealed class GridCombineOptions
{
    public CombineOperation Operation { get; set; } = CombineOperation.Subtract;
    public ValidMaskRule MaskRule { get; set; } = ValidMaskRule.BothValid;
    public DiffPresentation Presentation { get; set; } = DiffPresentation.Signed;
    public double? ClipMin { get; set; }
    public double? ClipMax { get; set; }
    public bool SymmetricVRangeForSigned { get; set; } = true;
}
