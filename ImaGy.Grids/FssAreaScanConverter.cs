using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace ImaGy.Grids;

/// <summary>FSS areascan CSV (롱형/와이드) 및 Interpolation 형식 변환 — Python myFSS_Conv_Peak4.py 대응.</summary>
public static class FssAreaScanConverter
{
    private static readonly string[] EncodingCandidates = ["utf-8", "utf-8", "cp949", "latin1"];

    public enum CsvShapeKind { Long, Wide }

    public sealed record LayoutScanResult(int SkipRows, string EncodingWebName, CsvShapeKind Shape);

    public sealed record LongPeekResult(int RowCount, int InferredWidth, int InferredHeight, string? ErrorMessage);

    public static string ColumnPatternForId(int idNum) => $@"\(ID:{idNum}\)(?!\d)";

    public static int FindColumnIndexById(IReadOnlyList<string> headers, int idNum)
    {
        var rx = new Regex(ColumnPatternForId(idNum), RegexOptions.CultureInvariant);
        for (var i = 0; i < headers.Count; i++)
        {
            if (rx.IsMatch(headers[i] ?? ""))
                return i;
        }
        throw new InvalidOperationException($"(ID:{idNum}) 컬럼을 찾을 수 없습니다.");
    }

    public static LayoutScanResult DetectLayout(string filePath, int maxScan = 80)
    {
        Exception? lastDecode = null;
        for (var ei = 0; ei < EncodingCandidates.Length; ei++)
        {
            var encName = EncodingCandidates[ei];
            Encoding enc = encName switch
            {
                "utf-8" when ei == 0 => new UTF8Encoding(encoderShouldEmitUTF8Identifier: true),
                "utf-8" => new UTF8Encoding(false),
                "cp949" => Encoding.GetEncoding(949),
                _ => Encoding.GetEncoding(28591) // latin-1
            };

            try
            {
                using var sr = new StreamReader(filePath, enc, detectEncodingFromByteOrderMarks: ei == 0, bufferSize: 65536);
                for (var skip = 0; skip < maxScan; skip++)
                {
                    var line = sr.ReadLine();
                    if (line == null)
                        return new LayoutScanResult(0, enc.WebName, CsvShapeKind.Long);
                    if (!line.Contains(';', StringComparison.Ordinal))
                        continue;
                    if (line.Contains("(ID:65)", StringComparison.Ordinal)
                        || line.Contains("(ID:82)", StringComparison.Ordinal)
                        || line.Contains("(ID:66)", StringComparison.Ordinal))
                        return new LayoutScanResult(skip, enc.WebName, CsvShapeKind.Long);
                    if (line.Count(c => c == ';') >= 8)
                        return new LayoutScanResult(skip, enc.WebName, CsvShapeKind.Wide);
                }
                return new LayoutScanResult(0, enc.WebName, CsvShapeKind.Long);
            }
            catch (DecoderFallbackException ex)
            {
                lastDecode = ex;
            }
        }
        throw lastDecode ?? new InvalidOperationException("인코딩을 판별할 수 없습니다.");
    }

    public static List<string> ReadWideMetaLines(string filePath, int maxScan = 80)
    {
        var layout = DetectLayout(filePath, maxScan);
        var enc = Encoding.GetEncoding(layout.EncodingWebName);
        using var sr = new StreamReader(filePath, enc, detectEncodingFromByteOrderMarks: false);
        var meta = new List<string>();
        for (var i = 0; i < maxScan; i++)
        {
            var line = sr.ReadLine();
            if (line == null)
                return meta;
            if (!line.Contains(';', StringComparison.Ordinal))
            {
                meta.Add(line.Trim());
                continue;
            }
            if (line.Contains("(ID:65)", StringComparison.Ordinal)
                || line.Contains("(ID:82)", StringComparison.Ordinal)
                || line.Contains("(ID:66)", StringComparison.Ordinal))
                return meta;
            if (line.Count(c => c == ';') >= 8)
                return meta;
            meta.Add(line.Trim());
        }
        return meta;
    }

    public static string WideOutputStemFromMeta(IReadOnlyList<string> metaLines, string baseName)
    {
        if (metaLines.Count == 0)
            return $"{baseName}_Grid";
        var line0 = metaLines[0].Trim();
        var line1 = metaLines.Count > 1 ? metaLines[1].Trim() : "";

        var m = Regex.Match(line0, @"(?i)peak\s+position\s*(\d+)");
        if (m.Success && int.TryParse(m.Groups[1].Value, out var pk))
            return $"{baseName}_Distance{pk}";

        m = Regex.Match(line0, @"(?i)quality\s*(\d+)");
        if (m.Success && int.TryParse(m.Groups[1].Value, out var qk))
            return $"{baseName}_Quality{qk}";

        if (Regex.IsMatch(line0, @"(?i)intensity"))
            return $"{baseName}_Intensity";

        foreach (var ln in new[] { line0, line1 })
        {
            m = Regex.Match(ln, @"(?i)signal\s*id\s*:\s*(\d+)");
            if (!m.Success || !int.TryParse(m.Groups[1].Value, out var sid))
                continue;
            if (sid == 82)
                return $"{baseName}_Intensity";
            if (sid >= 256 && (sid - 256) % 8 == 0)
            {
                var k = (sid - 256) / 8 + 1;
                return $"{baseName}_Distance{k}";
            }
            if (sid >= 257 && (sid - 257) % 8 == 0)
            {
                var k = (sid - 257) / 8 + 1;
                return $"{baseName}_Quality{k}";
            }
        }
        return $"{baseName}_Grid";
    }

    public static (int width, int height) InferRasterSizeFromXY(double[] x, double[] y) =>
        InferRasterSizeFromXY(x.AsSpan(), y.AsSpan());

    private static (int width, int height) InferRasterSizeFromXY(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        if (x.Length != y.Length)
            throw new InvalidOperationException("X와 Y 길이가 다릅니다.");
        var n = x.Length;
        if (n < 2)
            throw new InvalidOperationException("데이터 행이 너무 적습니다.");

        int? wFromJumps = RowWidthFromCoordJumps(x);
        if (wFromJumps is { } wj && wj > 0 && n % wj == 0)
            return (wj, n / wj);

        wFromJumps = RowWidthFromCoordJumps(y);
        if (wFromJumps is { } wj2 && wj2 > 0 && n % wj2 == 0)
            return (wj2, n / wj2);

        var rx = MaxMinSpan(x);
        var ry = MaxMinSpan(y);
        var targetWh = ry > 1e-12 ? rx / ry : 1.0;
        (int w, int h)? best = null;
        double bestErr = double.PositiveInfinity;
        var lim = (int)Math.Sqrt(n) + 1;
        for (var d = 1; d < lim; d++)
        {
            if (n % d != 0)
                continue;
            var h = n / d;
            TryPair(d, h);
            TryPair(h, d);
        }
        if (best is { } wh)
            return wh;
        throw new InvalidOperationException("그리드 크기를 추론할 수 없습니다.");

        void TryPair(int w, int ht)
        {
            var err = Math.Abs(w / Math.Max(ht, 1e-12) - targetWh);
            if (err < bestErr)
            {
                bestErr = err;
                best = (w, ht);
            }
        }
    }

    private static int? RowWidthFromCoordJumps(ReadOnlySpan<double> coord)
    {
        if (coord.Length < 2)
            return null;
        var diffs = new List<double>(coord.Length - 1);
        for (var i = 0; i < coord.Length - 1; i++)
            diffs.Add(Math.Abs(coord[i + 1] - coord[i]));
        diffs.Sort();
        var med = diffs.Count > 0 ? diffs[diffs.Count / 2] : 0.0;
        var thr = Math.Max(med * 15.0, 0.5);
        var jumps = new List<int>();
        for (var i = 0; i < coord.Length - 1; i++)
        {
            if (Math.Abs(coord[i + 1] - coord[i]) > thr)
                jumps.Add(i);
        }
        if (jumps.Count == 0)
            return null;
        var n = coord.Length;
        if (jumps.Count == 1)
        {
            var w = jumps[0] + 1;
            if (w > 0 && n % w == 0)
                return w;
            return null;
        }
        var gaps = new List<int>();
        for (var i = 0; i < jumps.Count - 1; i++)
        {
            var g = jumps[i + 1] - jumps[i];
            if (g > 0)
                gaps.Add(g);
        }
        int w2;
        if (gaps.Count == 0)
            w2 = jumps[0] + 1;
        else
        {
            gaps.Sort();
            w2 = gaps[gaps.Count / 2];
        }
        if (w2 <= 0 || n % w2 != 0)
            return null;
        return w2;
    }

    private static double MaxMinSpan(ReadOnlySpan<double> a)
    {
        if (a.Length == 0)
            return 0;
        var min = a[0];
        var max = a[0];
        foreach (var v in a)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return max - min;
    }

    /// <summary>롱형 파일에서 (ID:65)(ID:66)로 그리드 크기만 미리 계산합니다.</summary>
    public static LongPeekResult PeekLongGrid(string filePath)
    {
        try
        {
            var layout = DetectLayout(filePath);
            if (layout.Shape == CsvShapeKind.Wide)
                return new LongPeekResult(0, 0, 0, "와이드 형식 파일입니다. X/Y 빈 설정은 롱형에만 적용됩니다.");

            var enc = Encoding.GetEncoding(layout.EncodingWebName);
            using var sr = new StreamReader(filePath, enc, detectEncodingFromByteOrderMarks: false);
            for (var i = 0; i < layout.SkipRows; i++)
                sr.ReadLine();
            var headerLine = sr.ReadLine();
            if (headerLine == null)
                return new LongPeekResult(0, 0, 0, "헤더가 없습니다.");
            var headers = SplitSemicolonRow(headerLine);
            var ix = FindColumnIndexById(headers, 65);
            var iy = FindColumnIndexById(headers, 66);
            var xs = new List<double>();
            var ys = new List<double>();
            while (true)
            {
                var line = sr.ReadLine();
                if (line == null)
                    break;
                var parts = SplitSemicolonRow(line);
                if (parts.Count <= Math.Max(ix, iy))
                    continue;
                if (!TryParseDoubleLoose(parts[ix], out var xv) || !TryParseDoubleLoose(parts[iy], out var yv))
                    continue;
                xs.Add(xv);
                ys.Add(yv);
            }
            if (xs.Count < 2)
                return new LongPeekResult(0, 0, 0, "좌표 데이터가 부족합니다.");
            var (w, h) = InferRasterSizeFromXY(xs.ToArray(), ys.ToArray());
            return new LongPeekResult(xs.Count, w, h, null);
        }
        catch (Exception ex)
        {
            return new LongPeekResult(0, 0, 0, ex.Message);
        }
    }

    private static List<string> SplitSemicolonRow(string line) =>
        line.Split(';', StringSplitOptions.None).Select(s => s.Trim()).ToList();

    private static bool TryParseDoubleLoose(string s, out double v)
    {
        s = s.Trim();
        if (string.IsNullOrEmpty(s))
        {
            v = double.NaN;
            return false;
        }
        return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v)
               || double.TryParse(s, NumberStyles.Float, CultureInfo.CurrentCulture, out v);
    }

    public sealed class RawSaveFlags
    {
        public bool IntensityCsv { get; init; } = true;
        public bool DistanceCsv { get; init; } = true;
        public bool QualityCsv { get; init; } = true;
        public bool IntensityPng { get; init; } = true;
        public bool DistancePng { get; init; }
        public bool QualityPng { get; init; }
    }

    public static void ConvertRaw(
        string filePath,
        int searchRangeX,
        int searchRangeY,
        int convertCount,
        int width,
        int height,
        int maxParallelism,
        RawSaveFlags flags,
        IProgress<string>? progress = null)
    {
        convertCount = Math.Clamp(convertCount, 1, 4);
        maxParallelism = Math.Max(1, maxParallelism);
        var layout = DetectLayout(filePath);
        var dir = Path.GetDirectoryName(filePath) ?? ".";
        var baseName = Path.GetFileNameWithoutExtension(filePath);
        var enc = Encoding.GetEncoding(layout.EncodingWebName);

        if (layout.Shape == CsvShapeKind.Wide)
        {
            progress?.Report("P:0");
            ConvertRawWide(filePath, layout, enc, dir, baseName, flags, progress);
            return;
        }

        var table = ReadLongTable(filePath, layout, enc, out var headers);
        var ixCol = FindColumnIndexById(headers, 65);
        var iyCol = FindColumnIndexById(headers, 66);
        var iInt = FindColumnIndexById(headers, 82);
        var dataNum = width * height;
        if (table.Count < dataNum)
        {
            var missing = dataNum - table.Count;
            var last = table[^1];
            for (var m = 0; m < missing; m++)
                table.Add((string[])last.Clone());
        }
        else if (table.Count > dataNum)
            table.RemoveRange(dataNum, table.Count - dataNum);

        var xFlat = ExtractColumnDoubles(table, ixCol);
        var yFlat = ExtractColumnDoubles(table, iyCol);
        var intFlat = ExtractColumnDoubles(table, iInt);
        var distances = new double[convertCount][];
        var qualities = new double[convertCount][];
        for (var i = 0; i < convertCount; i++)
        {
            var dId = 256 + i * 8;
            var qId = 257 + i * 8;
            distances[i] = ExtractColumnDoubles(table, FindColumnIndexById(headers, dId));
            qualities[i] = ExtractColumnDoubles(table, FindColumnIndexById(headers, qId));
        }

        var x = ReshapeRowMajor(xFlat, width, height);
        var y = ReshapeRowMajor(yFlat, width, height);
        var intensityOrigin = ReshapeRowMajor(intFlat, width, height);
        var distOrig = distances.Select(d => ReshapeRowMajor(d, width, height)).ToArray();
        var qualOrig = qualities.Select(q => ReshapeRowMajor(q, width, height)).ToArray();

        var minX = Min2D(x);
        var maxX = Max2D(x);
        var minY = Min2D(y);
        var maxY = Max2D(y);
        var xRes = width > 1 ? (maxX - minX) / (width - 1) : 0;
        var yRes = height > 1 ? (maxY - minY) / (height - 1) : 0;

        var intensity = NewFilled2D(height, width, double.PositiveInfinity);
        var resDist = Enumerable.Range(0, convertCount).Select(_ => NewFilled2D(height, width, double.PositiveInfinity)).ToArray();
        var resQual = Enumerable.Range(0, convertCount).Select(_ => NewFilled2D(height, width, double.PositiveInfinity)).ToArray();

        progress?.Report("P:0");
        var rowsDone = 0;
        var progressGate = new object();
        var lastReportedPct = -1;
        void ReportRowProgress()
        {
            var done = Interlocked.Increment(ref rowsDone);
            var pct = height <= 0 ? 100 : (int)Math.Clamp(Math.Floor(done * 99.0 / height), 0, 99);
            lock (progressGate)
            {
                if (pct <= lastReportedPct)
                    return;
                lastReportedPct = pct;
            }
            progress?.Report($"P:{pct}");
        }

        var po = new ParallelOptions { MaxDegreeOfParallelism = maxParallelism };
        Parallel.For(0, height, po, iy =>
        {
            for (var ix = 0; ix < width; ix++)
            {
                var rx = minX + ix * xRes;
                var ry = minY + iy * yRes;
                var minDist = double.PositiveInfinity;
                var bestTx = -1;
                var bestTy = -1;
                for (var dy = -searchRangeY; dy <= searchRangeY; dy++)
                {
                    for (var dx = -searchRangeX; dx <= searchRangeX; dx++)
                    {
                        var tx = ix + dx;
                        var ty = iy + dy;
                        if ((uint)tx >= (uint)width || (uint)ty >= (uint)height)
                            continue;
                        var d = (rx - x[ty, tx]) * (rx - x[ty, tx]) + (ry - y[ty, tx]) * (ry - y[ty, tx]);
                        if (d < minDist)
                        {
                            minDist = d;
                            bestTx = tx;
                            bestTy = ty;
                        }
                    }
                }
                if (bestTx < 0)
                    continue;
                intensity[iy, ix] = intensityOrigin[bestTy, bestTx];
                for (var j = 0; j < convertCount; j++)
                {
                    resDist[j][iy, ix] = distOrig[j][bestTy, bestTx];
                    resQual[j][iy, ix] = qualOrig[j][bestTy, bestTx];
                }
            }
            ReportRowProgress();
        });
        progress?.Report("P:99");
        progress?.Report("S:CSV·PNG 저장 중…");

        if (flags.IntensityCsv)
            SaveCsvComma(Path.Combine(dir, $"{baseName}_Intensity.csv"), intensity);
        for (var j = 0; j < convertCount; j++)
        {
            if (flags.DistanceCsv)
                SaveCsvComma(Path.Combine(dir, $"{baseName}_Distance{j + 1}.csv"), resDist[j]);
            if (flags.QualityCsv)
                SaveCsvComma(Path.Combine(dir, $"{baseName}_Quality{j + 1}.csv"), resQual[j]);
        }

        if (flags.IntensityPng)
            TrySaveHeatmap(Path.Combine(dir, $"{baseName}_Intensity.png"), intensity, $"{baseName}  {width}×{height} (RAW)");
        if (flags.DistancePng)
        {
            for (var j = 0; j < convertCount; j++)
                TrySaveHeatmap(Path.Combine(dir, $"{baseName}_Distance{j + 1}.png"), resDist[j], $"{baseName} Distance{j + 1}  {width}×{height}");
        }
        if (flags.QualityPng)
        {
            for (var j = 0; j < convertCount; j++)
                TrySaveHeatmap(Path.Combine(dir, $"{baseName}_Quality{j + 1}.png"), resQual[j], $"{baseName} Quality{j + 1}  {width}×{height}");
        }
        progress?.Report("P:100");
    }

    private static void ConvertRawWide(
        string filePath,
        LayoutScanResult layout,
        Encoding enc,
        string dir,
        string baseName,
        RawSaveFlags flags,
        IProgress<string>? progress)
    {
        var grid = ReadWideNumericGrid(filePath, layout, enc);
        if (grid.GetLength(0) < 1 || grid.GetLength(1) < 1)
            throw new InvalidOperationException("와이드 형식인데 유효한 숫자 그리드가 없습니다.");
        var height = grid.GetLength(0);
        var width = grid.GetLength(1);
        var meta = ReadWideMetaLines(filePath);
        var stem = WideOutputStemFromMeta(meta, baseName);
        var wantCsv = flags.IntensityCsv || flags.DistanceCsv || flags.QualityCsv;
        var anyPng = flags.IntensityPng || flags.DistancePng || flags.QualityPng;
        if (wantCsv || anyPng)
        {
            var csvPath = Path.Combine(dir, $"{stem}.csv");
            SaveCsvComma(csvPath, grid);
            progress?.Report($"와이드 {width}×{height} CSV 저장");
        }
        if (anyPng)
            TrySaveHeatmap(Path.Combine(dir, $"{stem}.png"), grid, $"{stem}  가로{width}×세로{height}");
        progress?.Report("P:100");
    }

    private static double[,] ReadWideNumericGrid(string filePath, LayoutScanResult layout, Encoding enc)
    {
        using var sr = new StreamReader(filePath, enc, detectEncodingFromByteOrderMarks: false);
        for (var i = 0; i < layout.SkipRows; i++)
            sr.ReadLine();
        var rows = new List<double[]>();
        while (true)
        {
            var line = sr.ReadLine();
            if (line == null)
                break;
            var parts = SplitSemicolonRow(line);
            var nums = parts.Select(p => TryParseDoubleLoose(p, out var v) ? v : double.NaN).ToArray();
            if (nums.Length == 0 || nums.All(double.IsNaN))
                continue;
            rows.Add(nums);
        }
        if (rows.Count == 0)
            return new double[0, 0];
        var ncol = rows.Max(r => r.Length);
        var h = rows.Count;
        var w = ncol;
        var m = new double[h, w];
        for (var r = 0; r < h; r++)
        {
            var row = rows[r];
            for (var c = 0; c < row.Length && c < w; c++)
                m[r, c] = row[c];
        }
        return m;
    }

    private static List<string[]> ReadLongTable(string filePath, LayoutScanResult layout, Encoding enc, out List<string> headers)
    {
        using var sr = new StreamReader(filePath, enc, detectEncodingFromByteOrderMarks: false);
        for (var i = 0; i < layout.SkipRows; i++)
            sr.ReadLine();
        var headerLine = sr.ReadLine() ?? throw new InvalidOperationException("헤더가 없습니다.");
        headers = SplitSemicolonRow(headerLine);
        var rows = new List<string[]>();
        while (true)
        {
            var line = sr.ReadLine();
            if (line == null)
                break;
            var parts = SplitSemicolonRow(line);
            if (parts.Count < headers.Count)
            {
                while (parts.Count < headers.Count)
                    parts.Add("");
            }
            rows.Add(parts.ToArray());
        }
        return rows;
    }

    private static double[] ExtractColumnDoubles(List<string[]> table, int col)
    {
        var a = new double[table.Count];
        for (var i = 0; i < table.Count; i++)
        {
            var s = col < table[i].Length ? table[i][col] : "";
            _ = TryParseDoubleLoose(s, out a[i]);
        }
        return a;
    }

    private static double[,] ReshapeRowMajor(double[] flat, int width, int height)
    {
        var m = new double[height, width];
        var idx = 0;
        for (var row = 0; row < height; row++)
        {
            for (var col = 0; col < width; col++)
                m[row, col] = flat[idx++];
        }
        return m;
    }

    private static double[,] NewFilled2D(int h, int w, double fill)
    {
        var m = new double[h, w];
        for (var i = 0; i < h; i++)
        {
            for (var j = 0; j < w; j++)
                m[i, j] = fill;
        }
        return m;
    }

    private static double Min2D(double[,] a)
    {
        var v = double.PositiveInfinity;
        foreach (var x in a)
        {
            if (x < v) v = x;
        }
        return v;
    }

    private static double Max2D(double[,] a)
    {
        var v = double.NegativeInfinity;
        foreach (var x in a)
        {
            if (x > v) v = x;
        }
        return v;
    }

    private static void SaveCsvComma(string path, double[,] z)
    {
        var h = z.GetLength(0);
        var w = z.GetLength(1);
        using var tw = new StreamWriter(path, false, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        for (var r = 0; r < h; r++)
        {
            for (var c = 0; c < w; c++)
            {
                if (c > 0) tw.Write(',');
                tw.Write(z[r, c].ToString("G17", CultureInfo.InvariantCulture));
            }
            tw.WriteLine();
        }
    }

    private static void TrySaveHeatmap(string path, double[,] z, string title)
    {
        var (lo, hi) = RobustFiniteMinMax(z);
        if (!double.IsFinite(lo))
        {
            lo = 0;
            hi = 1;
        }
        if (Math.Abs(hi - lo) < 1e-30)
        {
            lo -= 1;
            hi += 1;
        }
        var rows = z.GetLength(0);
        var cols = z.GetLength(1);
        var wPx = Math.Clamp(cols * 4, 400, 2000);
        var hPx = Math.Clamp(rows * 4, 300, 2000);
        GridScottPlotExporter.SaveMatplotlibStyleHeatmap(path, z, lo, hi, title, "VIRIDIS", wPx, hPx);
    }

    private static (double lo, double hi) RobustFiniteMinMax(double[,] z)
    {
        var lo = double.PositiveInfinity;
        var hi = double.NegativeInfinity;
        foreach (var v in z)
        {
            if (!double.IsFinite(v))
                continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return (lo, hi);
    }

    public sealed class InterpSaveFlags
    {
        public bool DistanceCsv { get; init; } = true;
        public bool QualityCsv { get; init; } = true;
        public bool DistancePng { get; init; } = true;
        public bool QualityPng { get; init; }
    }

    public static void ConvertInterp(string filePath, int convertCount, InterpSaveFlags flags, IProgress<string>? progress = null)
    {
        convertCount = Math.Clamp(convertCount, 1, 4);
        progress?.Report("P:0");
        var dir = Path.GetDirectoryName(filePath) ?? ".";
        var baseName = Path.GetFileNameWithoutExtension(filePath);
        double[,]? arr = null;
        foreach (var encName in new[] { "utf-8", "utf-8", "cp949", "latin1" })
        {
            Encoding enc = encName == "utf-8" && arr == null
                ? new UTF8Encoding(true)
                : encName == "utf-8"
                    ? new UTF8Encoding(false)
                    : encName == "cp949"
                        ? Encoding.GetEncoding(949)
                        : Encoding.GetEncoding(28591);
            try
            {
                arr = ReadInterpMatrix(filePath, enc);
                break;
            }
            catch (DecoderFallbackException) { }
        }
        progress?.Report("P:30");
        arr ??= ReadInterpMatrix(filePath, Encoding.GetEncoding(28591));
        if (arr.Length == 0)
            throw new InvalidOperationException("Interp: 숫자 데이터가 없습니다.");

        var n = arr.GetLength(0);
        var wloc = arr.GetLength(1);
        var npairsMax = convertCount;

        void SavePairs(double[][,] chunks, int npairs, int h)
        {
            for (var i = 0; i < npairs; i++)
            {
                if (flags.DistanceCsv)
                    SaveCsvComma(Path.Combine(dir, $"{baseName}_Distance{i + 1}.csv"), chunks[2 * i]);
                if (flags.QualityCsv)
                    SaveCsvComma(Path.Combine(dir, $"{baseName}_Quality{i + 1}.csv"), chunks[2 * i + 1]);
            }
            if (flags.DistancePng)
            {
                for (var i = 0; i < npairs; i++)
                    TrySaveHeatmap(Path.Combine(dir, $"{baseName}_Distance{i + 1}.png"), chunks[2 * i],
                        $"{baseName} Distance{i + 1} 가로{wloc}×세로{h}");
            }
            if (flags.QualityPng)
            {
                for (var i = 0; i < npairs; i++)
                    TrySaveHeatmap(Path.Combine(dir, $"{baseName}_Quality{i + 1}.png"), chunks[2 * i + 1],
                        $"{baseName} Quality{i + 1} 가로{wloc}×세로{h}");
            }
            progress?.Report($"Interp {npairs}쌍 저장 완료");
            progress?.Report("P:100");
        }

        for (var npairs = npairsMax; npairs >= 1; npairs--)
        {
            if (n % (2 * npairs) != 0)
                continue;
            var h = n / (2 * npairs);
            var chunks = new double[2 * npairs][,];
            for (var i = 0; i < 2 * npairs; i++)
            {
                chunks[i] = SubMatrix(arr, i * h, (i + 1) * h, 0, wloc);
            }
            if (chunks.All(c => c.GetLength(0) == h && c.GetLength(1) == wloc && AnyFinite(c)))
            {
                SavePairs(chunks, npairs, h);
                return;
            }
        }

        if (n % 2 == 0)
        {
            var h = n / 2;
            var top = SubMatrix(arr, 0, h, 0, wloc);
            var bot = SubMatrix(arr, h, n, 0, wloc);
            if (top.GetLength(0) >= 1 && top.GetLength(0) == bot.GetLength(0))
            {
                SavePairs(new[] { top, bot }, 1, top.GetLength(0));
                return;
            }
        }

        var fallback = Path.Combine(dir, $"{baseName}_Conv.csv");
        SaveInterpRawFallback(fallback, arr);
        progress?.Report("_Conv.csv (피크+퀄 블록 미검출)");
        progress?.Report("P:100");
    }

    private static bool AnyFinite(double[,] c)
    {
        foreach (var v in c)
        {
            if (double.IsFinite(v))
                return true;
        }
        return false;
    }

    private static double[,] SubMatrix(double[,] a, int r0, int r1, int c0, int c1)
    {
        var h = r1 - r0;
        var w = c1 - c0;
        var m = new double[h, w];
        for (var i = 0; i < h; i++)
        {
            for (var j = 0; j < w; j++)
                m[i, j] = a[r0 + i, c0 + j];
        }
        return m;
    }

    /// <summary>pandas read_csv(skiprows=4, header=None) 기본 구분자 쉼표 — 첫 셀은 세미콜론으로 이어진 값 묶음.</summary>
    private static double[,] ReadInterpMatrix(string filePath, Encoding enc)
    {
        using var sr = new StreamReader(filePath, enc, detectEncodingFromByteOrderMarks: false);
        for (var s = 0; s < 4; s++)
            sr.ReadLine();
        var rows = new List<List<double>>();
        while (true)
        {
            var line = sr.ReadLine();
            if (line == null)
                break;
            var commaParts = SplitCsvSimple(line);
            if (commaParts.Count == 0)
                continue;
            var rowVals = new List<string>();
            rowVals.AddRange(commaParts[0].Split(';', StringSplitOptions.None));
            for (var k = 1; k < commaParts.Count; k++)
                rowVals.Add(commaParts[k]);
            var nums = rowVals.Select(s => TryParseDoubleLoose(s.Trim(), out var v) ? v : double.NaN).ToList();
            if (nums.Count > 0 && !nums.All(double.IsNaN))
                rows.Add(nums);
        }
        if (rows.Count == 0)
            return new double[0, 0];
        var ncol = rows.Max(r => r.Count);
        var n = rows.Count;
        var m = new double[n, ncol];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < rows[i].Count && j < ncol; j++)
                m[i, j] = rows[i][j];
        }
        return RemoveAllNanRows(m);
    }

    /// <summary>따옴표 없는 CSV 한 줄을 쉼표로 분리 (Interpolation보내기용).</summary>
    private static List<string> SplitCsvSimple(string line) =>
        line.Split(',', StringSplitOptions.None).Select(s => s.Trim()).ToList();

    private static double[,] RemoveAllNanRows(double[,] m)
    {
        var n = m.GetLength(0);
        var w = m.GetLength(1);
        var keep = new List<int>();
        for (var i = 0; i < n; i++)
        {
            var allNan = true;
            for (var j = 0; j < w; j++)
            {
                if (!double.IsNaN(m[i, j]))
                {
                    allNan = false;
                    break;
                }
            }
            if (!allNan)
                keep.Add(i);
        }
        var m2 = new double[keep.Count, w];
        for (var i = 0; i < keep.Count; i++)
        {
            for (var j = 0; j < w; j++)
                m2[i, j] = m[keep[i], j];
        }
        return m2;
    }

    private static void SaveInterpRawFallback(string path, double[,] arr)
    {
        var n = arr.GetLength(0);
        var w = arr.GetLength(1);
        using var tw = new StreamWriter(path, false, new UTF8Encoding(true));
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < w; j++)
            {
                if (j > 0) tw.Write(',');
                tw.Write(arr[i, j].ToString("G17", CultureInfo.InvariantCulture));
            }
            tw.WriteLine();
        }
    }
}
