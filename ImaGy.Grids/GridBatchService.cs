using System.Text;

namespace ImaGy.Grids;

public sealed class GridBatchOptions
{
    public GridAlignmentOptions Alignment { get; set; } = new();
    public GridPreprocessOptions PreprocessA { get; set; } = new();
    public GridPreprocessOptions PreprocessB { get; set; } = new();
    public GridCombineOptions Combine { get; set; } = new();
    public GridVisualizationOptions Visualization { get; set; } = new();
    public string InputColormapId { get; set; } = "Turbo";
    public string DiffColormapId { get; set; } = "Turbo";
    public int? HeatmapMaxDim { get; set; } = 2048;
    public bool WriteHeatmaps { get; set; } = true;
    public bool WriteHistogramPng { get; set; } = false;
    public bool WriteMaskedDiffHeatmap { get; set; } = false;
    public GridRoiCatalog? Catalog { get; set; }
    public double? StatsValueMin { get; set; }
    public double? StatsValueMax { get; set; }
}

public static class GridBatchService
{
    /// <summary>Pair files where pathB equals pathA with first occurrence of "Distance1" replaced by "Distance2".</summary>
    public static IReadOnlyList<(string PathA, string PathB, string Label)> FindDistancePairs(string folder)
    {
        if (!Directory.Exists(folder)) return Array.Empty<(string, string, string)>();
        var list = new List<(string, string, string)>();
        foreach (var pathA in Directory.EnumerateFiles(folder, "*.csv", SearchOption.TopDirectoryOnly))
        {
            if (pathA.IndexOf("Distance1", StringComparison.OrdinalIgnoreCase) < 0) continue;
            var pathB = pathA.Replace("Distance1", "Distance2", StringComparison.OrdinalIgnoreCase);
            if (!File.Exists(pathB)) continue;
            list.Add((pathA, pathB, Path.GetFileNameWithoutExtension(pathA)));
        }
        return list;
    }

    public static void RunPair(string pathA, string pathB, string outputDir, GridBatchOptions opt, Action<string>? log = null)
    {
        Directory.CreateDirectory(outputDir);
        void L(string m) { log?.Invoke(m); }
        L($"Load {pathA}");
        var rawA = CsvGridReader.ReadFromFile(pathA);
        L($"Load {pathB}");
        var rawB = CsvGridReader.ReadFromFile(pathB);
        var result = GridPipeline.Run(rawA, rawB, opt.Alignment, opt.PreprocessA, opt.PreprocessB, opt.Combine);
        var diff = result.Diff;
        string baseName = Path.GetFileNameWithoutExtension(pathA) + "__" + Path.GetFileNameWithoutExtension(pathB);
        string diffCsv = Path.Combine(outputDir, baseName + "_diff.csv");
        CsvGridReader.WriteToFile(diff, diffCsv);
        L($"Wrote {diffCsv}");

        var vis = opt.Visualization;
        var (dmin, dmax) = GridCombineService.GetDisplayRange(diff, opt.Combine);
        if (opt.WriteHeatmaps)
        {
            var (aLo, aHi) = GridVisualizationService.GetNormalizeRange(result.ProcessedA, vis);
            var (bLo, bHi) = GridVisualizationService.GetNormalizeRange(result.ProcessedB, vis);
            GridVisualizationService.SaveHeatmapPng(result.ProcessedA, Path.Combine(outputDir, $"{baseName}_A_cmap-{opt.InputColormapId}.png"), aLo, aHi, opt.InputColormapId, vis, opt.HeatmapMaxDim);
            GridVisualizationService.SaveHeatmapPng(result.ProcessedB, Path.Combine(outputDir, $"{baseName}_B_cmap-{opt.InputColormapId}.png"), bLo, bHi, opt.InputColormapId, vis, opt.HeatmapMaxDim);
            GridVisualizationService.SaveHeatmapPng(diff, Path.Combine(outputDir, $"{baseName}_Diff_cmap-{opt.DiffColormapId}.png"), dmin, dmax, opt.DiffColormapId, vis, opt.HeatmapMaxDim);
        }

        var statSb = new StringBuilder();
        statSb.AppendLine($"pairA={pathA}");
        statSb.AppendLine($"pairB={pathB}");
        statSb.AppendLine($"diffShape={diff.Rows}x{diff.Cols}");
        var diffStats = GridStatisticsService.Compute(diff, null, opt.StatsValueMin, opt.StatsValueMax);
        statSb.AppendLine($"diff_global: count={diffStats.Count} mean={diffStats.Mean} std={diffStats.Std} min={diffStats.Min} max={diffStats.Max} median={diffStats.Median}");

            if (opt.Catalog != null)
            {
                opt.Catalog.ValidateAgainstGrid(diff);
                var roiSummaryPath = Path.Combine(outputDir, baseName + "_roi_summary.csv");
                if (!File.Exists(roiSummaryPath))
                    File.WriteAllText(roiSummaryPath, "roiId,roiName,maskTrueCells,validFiniteCount,mean,std,min,max,median,p25,p75" + Environment.NewLine);
                foreach (var roi in opt.Catalog.EnumerateDefinitions())
                {
                    var mask = GridRoiMaskBuilder.BuildMask(diff.Rows, diff.Cols, roi);
                    int maskTrue = CountMaskTrue(mask);
                    var st = GridStatisticsService.Compute(diff, mask, opt.StatsValueMin, opt.StatsValueMax);
                    string roiId = string.IsNullOrWhiteSpace(roi.Id) ? roi.Name : roi.Id;
                    statSb.AppendLine($"roi={roiId}/{roi.Name} maskTrueCells={maskTrue} validFiniteCount={st.Count} mean={st.Mean} std={st.Std} min={st.Min} max={st.Max} median={st.Median}");
                    string roiCsvLine = $"{Escape(roiId)},{Escape(roi.Name)},{maskTrue},{st.Count},{st.Mean},{st.Std},{st.Min},{st.Max},{st.Median},{st.P25},{st.P75}";
                    File.AppendAllText(roiSummaryPath, roiCsvLine + Environment.NewLine);
                if (opt.WriteMaskedDiffHeatmap)
                {
                    var masked = ApplyMaskToGrid(diff, mask);
                    GridVisualizationService.SaveHeatmapPng(masked, Path.Combine(outputDir, $"{baseName}_Diff_masked_{roi.Name}_cmap-{opt.DiffColormapId}.png"), dmin, dmax, opt.DiffColormapId, vis, opt.HeatmapMaxDim);
                }
                if (opt.WriteHistogramPng)
                {
                    var hist = GridStatisticsService.ComputeHistogram(diff, mask, opt.StatsValueMin, opt.StatsValueMax, 64);
                    GridScottPlotExporter.SaveHistogramPng(Path.Combine(outputDir, $"{baseName}_hist_{roi.Name}.png"), hist, roi.Name);
                }
            }
        }

        File.WriteAllText(Path.Combine(outputDir, baseName + "_stat.txt"), statSb.ToString());
        L($"Wrote stat.txt");
    }

    private static FloatGrid ApplyMaskToGrid(FloatGrid diff, bool[] mask)
    {
        var d = (double[])diff.Data.Clone();
        for (int i = 0; i < d.Length; i++)
        {
            if (i >= mask.Length || !mask[i]) d[i] = double.NaN;
        }
        return new FloatGrid(diff.Rows, diff.Cols, d, (bool[])diff.OriginallyFinite.Clone());
    }

    public static void RunFolderParallel(string folder, string outputRoot, GridBatchOptions opt, int maxDegreeOfParallelism = 0)
    {
        var pairs = FindDistancePairs(folder);
        if (pairs.Count == 0) return;
        var po = new ParallelOptions();
        if (maxDegreeOfParallelism > 0) po.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        Parallel.ForEach(pairs, po, pair =>
        {
            string sub = Path.Combine(outputRoot, pair.Label);
            RunPair(pair.PathA, pair.PathB, sub, opt, null);
        });
    }

    /// <summary>Apply catalog stats to existing diff CSV files in a folder.</summary>
    public static void SummarizeDiffCsvsWithCatalog(string diffCsvFolder, GridRoiCatalog catalog, string outputCsvPath, Action<string>? log = null)
    {
        var lines = new List<string> { "file,roiId,roiName,maskTrueCells,validFiniteCount,mean,std,min,max,median,p25,p75" };
        if (!Directory.Exists(diffCsvFolder))
            throw new DirectoryNotFoundException(diffCsvFolder);
        foreach (var path in Directory.EnumerateFiles(diffCsvFolder, "*_diff.csv", SearchOption.AllDirectories))
        {
            var grid = CsvGridReader.ReadFromFile(path);
            catalog.ValidateAgainstGrid(grid);
            foreach (var roi in catalog.EnumerateDefinitions())
            {
                var mask = GridRoiMaskBuilder.BuildMask(grid.Rows, grid.Cols, roi);
                int maskTrue = CountMaskTrue(mask);
                var st = GridStatisticsService.Compute(grid, mask, null, null);
                string roiId = string.IsNullOrWhiteSpace(roi.Id) ? roi.Name : roi.Id;
                lines.Add($"{Escape(Path.GetFileName(path))},{Escape(roiId)},{Escape(roi.Name)},{maskTrue},{st.Count},{st.Mean},{st.Std},{st.Min},{st.Max},{st.Median},{st.P25},{st.P75}");
            }
        }
        File.WriteAllLines(outputCsvPath, lines);
        log?.Invoke($"Wrote {outputCsvPath}");
    }

    private static string Escape(string s)
    {
        if (s.Contains(',') || s.Contains('"'))
            return "\"" + s.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
        return s;
    }

    /// <summary>ROI 마스크에서 true인 격자 셀 개수(도형 면적). 통계의 validFiniteCount와 다를 수 있음(ROI 내 NaN 등).</summary>
    private static int CountMaskTrue(bool[] mask)
    {
        int n = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i]) n++;
        }
        return n;
    }
}
