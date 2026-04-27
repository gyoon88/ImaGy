using ImaGy.Grids;

int Usage()
{
    Console.WriteLine("""
        ImaGy.GridCli — CSV grid batch & diff summary

        batch --folder <dir> [--out <dir>] [--catalog <roi.json>]
        summarize --folder <dirWithDiffCsv> --catalog <roi.json> --out <summary.csv>
        pair --a <a.csv> --b <b.csv> --out <dir>
        """);
    return 1;
}

try
{
    if (args.Length < 1)
        return Usage();

    var cmd = args[0].ToLowerInvariant();
    if (cmd == "batch")
    {
        string? folder = null, outDir = null, catalog = null;
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--folder" && i + 1 < args.Length) folder = args[++i];
            else if (args[i] == "--out" && i + 1 < args.Length) outDir = args[++i];
            else if (args[i] == "--catalog" && i + 1 < args.Length) catalog = args[++i];
        }
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return Usage();
        outDir ??= Path.Combine(folder, "_batch_cli_out");
        var opt = new GridBatchOptions();
        if (!string.IsNullOrEmpty(catalog) && File.Exists(catalog))
            opt.Catalog = GridRoiCatalog.Load(catalog);
        GridBatchService.RunFolderParallel(folder, outDir, opt);
        Console.WriteLine($"Done → {outDir}");
        return 0;
    }
    if (cmd == "summarize")
    {
        string? folder = null, catalog = null, outp = null;
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--folder" && i + 1 < args.Length) folder = args[++i];
            else if (args[i] == "--catalog" && i + 1 < args.Length) catalog = args[++i];
            else if (args[i] == "--out" && i + 1 < args.Length) outp = args[++i];
        }
        if (string.IsNullOrEmpty(folder) || string.IsNullOrEmpty(catalog) || string.IsNullOrEmpty(outp))
            return Usage();
        var cat = GridRoiCatalog.Load(catalog);
        GridBatchService.SummarizeDiffCsvsWithCatalog(folder, cat, outp, Console.WriteLine);
        return 0;
    }
    if (cmd == "pair")
    {
        string? a = null, b = null, outp = null;
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--a" && i + 1 < args.Length) a = args[++i];
            else if (args[i] == "--b" && i + 1 < args.Length) b = args[++i];
            else if (args[i] == "--out" && i + 1 < args.Length) outp = args[++i];
        }
        if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b) || string.IsNullOrEmpty(outp))
            return Usage();
        Directory.CreateDirectory(outp);
        GridBatchService.RunPair(a, b, outp, new GridBatchOptions(), Console.WriteLine);
        return 0;
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex.Message);
    return 2;
}

return Usage();
