namespace ImaGy.Grids;

public enum RoiKind
{
    Rectangle,
    Triangle,
    Circle
}

public sealed class GridRoiDefinition
{
    public string Name { get; set; } = "roi";
    public RoiKind Kind { get; set; }
    /// <summary>Rectangle: top-left column (x).</summary>
    public int X { get; set; }
    /// <summary>Rectangle: top-left row (y).</summary>
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public int C0 { get; set; }
    public int R0 { get; set; }
    public int C1 { get; set; }
    public int R1 { get; set; }
    public int C2 { get; set; }
    public int R2 { get; set; }
    public int CenterCol { get; set; }
    public int CenterRow { get; set; }
    public double RadiusPixels { get; set; }
}
