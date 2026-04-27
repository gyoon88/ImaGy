using System.Text.Json;
using System.Text.Json.Serialization;

namespace ImaGy.Grids;

public sealed class RoiCatalogEntryDto
{
    public string Kind { get; set; } = "Rectangle";
    public string Name { get; set; } = "roi";
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

public sealed class GridRoiCatalog
{
    public string Schema { get; set; } = "imagy.roi.catalog.v1";
    public int ReferenceRows { get; set; }
    public int ReferenceCols { get; set; }
    public int ReferencePairIndex { get; set; }
    public List<RoiCatalogEntryDto> Rois { get; set; } = new();

    private static readonly JsonSerializerOptions SaveOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    private static readonly JsonSerializerOptions LoadOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public static GridRoiCatalog Load(string path) =>
        JsonSerializer.Deserialize<GridRoiCatalog>(File.ReadAllText(path), LoadOptions)
        ?? throw new InvalidDataException("Invalid catalog JSON.");

    public void Save(string path) =>
        File.WriteAllText(path, JsonSerializer.Serialize(this, SaveOptions));

    public IEnumerable<GridRoiDefinition> EnumerateDefinitions() =>
        Rois.Select(ToDefinition);

    public static GridRoiDefinition ToDefinition(RoiCatalogEntryDto e)
    {
        var kind = Enum.TryParse<RoiKind>(e.Kind, true, out var k) ? k : RoiKind.Rectangle;
        return new GridRoiDefinition
        {
            Kind = kind,
            Name = e.Name,
            X = e.X,
            Y = e.Y,
            Width = e.Width,
            Height = e.Height,
            C0 = e.C0 ?? 0,
            R0 = e.R0 ?? 0,
            C1 = e.C1 ?? 0,
            R1 = e.R1 ?? 0,
            C2 = e.C2 ?? 0,
            R2 = e.R2 ?? 0,
            CenterCol = e.CenterCol ?? 0,
            CenterRow = e.CenterRow ?? 0,
            RadiusPixels = e.RadiusPixels ?? 0
        };
    }

    public void ValidateAgainstGrid(FloatGrid grid)
    {
        if (grid.Rows != ReferenceRows || grid.Cols != ReferenceCols)
            throw new InvalidOperationException(
                $"Catalog reference ({ReferenceRows}x{ReferenceCols}) does not match grid ({grid.Rows}x{grid.Cols}).");
    }
}
