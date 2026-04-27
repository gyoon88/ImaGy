using OpenCvSharp;

namespace ImaGy.Grids;

public static class GridColormapCatalog
{
    public static IReadOnlyList<(string Id, string Label)> InputColormaps { get; } =
    [
        ("Turbo", "Turbo"),
        ("Viridis", "Viridis"),
        ("Jet", "Jet"),
        ("Hot", "Hot"),
        ("Parula", "Parula (alias: Turbo)"),
        ("Gray", "Gray"),
    ];

    public static IReadOnlyList<(string Id, string Label)> DiffColormaps { get; } =
    [
        ("Turbo", "Turbo"),
        ("Jet", "Jet"),
        ("Cool", "Cool"),
        ("Gray", "Gray"),
    ];

    public static ColormapTypes Resolve(string id)
    {
        var k = id.Trim();
        return k.ToUpperInvariant() switch
        {
            "TURBO" => ColormapTypes.Turbo,
            "VIRIDIS" => ColormapTypes.Viridis,
            "JET" => ColormapTypes.Jet,
            "HOT" => ColormapTypes.Hot,
            "PARULA" => ColormapTypes.Turbo,
            "COOL" => ColormapTypes.Cool,
            "GRAY" => ColormapTypes.Turbo,
            _ => ColormapTypes.Turbo
        };
    }

    public static bool IsIntensityOnly(string id) =>
        id.Trim().Equals("Gray", StringComparison.OrdinalIgnoreCase);
}
