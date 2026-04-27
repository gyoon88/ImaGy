using System.Text.Json;
using System.Text.Json.Serialization;

namespace ImaGy.Grids;

public static class GridRoiJsonIO
{
    private static readonly JsonSerializerOptions SaveOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private static readonly JsonSerializerOptions LoadOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public sealed class SingleRoiFile
    {
        public string Schema { get; set; } = "imagy.roi.v1";
        public int ReferenceRows { get; set; }
        public int ReferenceCols { get; set; }
        public string Kind { get; set; } = "Rectangle";
        public string? Name { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int? C0 { get; set; }
        public int? R0 { get; set; }
        public int? C1 { get; set; }
        public int? R1 { get; set; }
        public int? C2 { get; set; }
        public int? R2 { get; set; }
        public int? CenterCol { get; set; }
        public int? CenterRow { get; set; }
        public double? RadiusPixels { get; set; }
    }

    public static void WriteSingleRoi(string path, int refRows, int refCols, GridRoiDefinition roi)
    {
        var dto = new SingleRoiFile
        {
            ReferenceRows = refRows,
            ReferenceCols = refCols,
            Name = roi.Name,
            Kind = roi.Kind.ToString(),
            X = roi.X,
            Y = roi.Y,
            Width = roi.Width,
            Height = roi.Height,
            C0 = roi.C0, R0 = roi.R0, C1 = roi.C1, R1 = roi.R1, C2 = roi.C2, R2 = roi.R2,
            CenterCol = roi.CenterCol,
            CenterRow = roi.CenterRow,
            RadiusPixels = roi.RadiusPixels
        };
        File.WriteAllText(path, JsonSerializer.Serialize(dto, SaveOptions));
    }

    public static GridRoiDefinition ReadSingleRoi(string path, out int refRows, out int refCols)
    {
        var dto = JsonSerializer.Deserialize<SingleRoiFile>(File.ReadAllText(path), LoadOptions)
                  ?? throw new InvalidDataException("Invalid ROI JSON.");
        refRows = dto.ReferenceRows;
        refCols = dto.ReferenceCols;
        var kind = Enum.TryParse<RoiKind>(dto.Kind, true, out var k) ? k : RoiKind.Rectangle;
        return new GridRoiDefinition
        {
            Name = dto.Name ?? "roi",
            Kind = kind,
            X = dto.X,
            Y = dto.Y,
            Width = dto.Width,
            Height = dto.Height,
            C0 = dto.C0 ?? 0,
            R0 = dto.R0 ?? 0,
            C1 = dto.C1 ?? 0,
            R1 = dto.R1 ?? 0,
            C2 = dto.C2 ?? 0,
            R2 = dto.R2 ?? 0,
            CenterCol = dto.CenterCol ?? 0,
            CenterRow = dto.CenterRow ?? 0,
            RadiusPixels = dto.RadiusPixels ?? 0
        };
    }

    public static void ValidateReferenceShape(int refRows, int refCols, FloatGrid grid)
    {
        if (grid.Rows != refRows || grid.Cols != refCols)
            throw new InvalidOperationException(
                $"Reference shape ({refRows}x{refCols}) does not match grid ({grid.Rows}x{grid.Cols}).");
    }
}
