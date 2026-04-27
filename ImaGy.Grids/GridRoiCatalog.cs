using System.Text.Json;
using System.Text.Json.Serialization;

namespace ImaGy.Grids;

public sealed class RoiCatalogEntryDto
{
    public string? Id { get; set; }
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
    /// <summary>ROI를 그릴 때 기준으로 삼은 격자 행 수. 0이면 JSON에 크기가 없거나 레거시이며, <see cref="ValidateAgainstGrid"/>는 실제 격자 크기와의 일치 검사를 생략합니다.</summary>
    public int ReferenceRows { get; set; }
    /// <summary>ROI를 그릴 때 기준으로 삼은 격자 열 수. 0이면 <see cref="ReferenceRows"/>와 같이 미지정으로 처리합니다.</summary>
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

    public static GridRoiCatalog Load(string path)
    {
        var c = JsonSerializer.Deserialize<GridRoiCatalog>(File.ReadAllText(path), LoadOptions)
            ?? throw new InvalidDataException("Invalid catalog JSON.");
        c.Rois ??= new List<RoiCatalogEntryDto>();
        return c;
    }

    public void Save(string path) =>
        File.WriteAllText(path, JsonSerializer.Serialize(this, SaveOptions));

    public IEnumerable<GridRoiDefinition> EnumerateDefinitions() =>
        Rois.Select(ToDefinition);

    public static GridRoiDefinition ToDefinition(RoiCatalogEntryDto e)
    {
        var kind = Enum.TryParse<RoiKind>(e.Kind, true, out var k) ? k : RoiKind.Rectangle;
        return new GridRoiDefinition
        {
            Id = e.Id ?? "",
            Kind = kind,
            Name = string.IsNullOrWhiteSpace(e.Name) ? (e.Id ?? "roi") : e.Name,
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

    /// <summary>
    /// 카탈로그가 기록한 기준 격자 크기와 <paramref name="grid"/>가 같은지 검사합니다.
    /// <see cref="ReferenceRows"/>와 <see cref="ReferenceCols"/>가 둘 다 0이면(파일에 크기 미기재·구버전 등) 검사를 건너뜁니다.
    /// </summary>
    public void ValidateAgainstGrid(FloatGrid grid)
    {
        if (ReferenceRows == 0 && ReferenceCols == 0)
            return;

        if (ReferenceRows <= 0 || ReferenceCols <= 0)
            throw new InvalidOperationException(
                $"Catalog reference dimensions are invalid ({ReferenceRows}x{ReferenceCols}). Use positive sizes, or (0,0) to skip the shape check.");

        if (grid.Rows != ReferenceRows || grid.Cols != ReferenceCols)
            throw new InvalidOperationException(
                $"Catalog reference ({ReferenceRows}x{ReferenceCols}) does not match grid ({grid.Rows}x{grid.Cols}).");
    }
}
