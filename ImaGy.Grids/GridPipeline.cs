namespace ImaGy.Grids;

public sealed class GridPipelineResult
{
    public FloatGrid AlignedA { get; init; } = null!;
    public FloatGrid AlignedB { get; init; } = null!;
    public FloatGrid ProcessedA { get; init; } = null!;
    public FloatGrid ProcessedB { get; init; } = null!;
    public FloatGrid Diff { get; init; } = null!;
}

public static class GridPipeline
{
    public static GridPipelineResult Run(
        FloatGrid rawA,
        FloatGrid rawB,
        GridAlignmentOptions align,
        GridPreprocessOptions preA,
        GridPreprocessOptions preB,
        GridCombineOptions combine)
    {
        var (a0, b0) = GridAlignmentService.Align(rawA, rawB, align);
        var procA = a0.Clone();
        var procB = b0.Clone();
        GridPreprocessService.Apply(procA, preA);
        GridPreprocessService.Apply(procB, preB);
        var diff = GridCombineService.Combine(procA, procB, combine);
        return new GridPipelineResult
        {
            AlignedA = a0,
            AlignedB = b0,
            ProcessedA = procA,
            ProcessedB = procB,
            Diff = diff
        };
    }
}
