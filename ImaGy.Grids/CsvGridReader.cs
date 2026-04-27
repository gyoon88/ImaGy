using System.Globalization;
using System.Text;

namespace ImaGy.Grids;

public static class CsvGridReader
{
    private static readonly string[] MissingTokens = { "", "NA", "N/A", "NAN", "NULL", "#N/A" };

    /// <summary>Reads a rectangular CSV of floats; ragged rows pad with NaN to max width.</summary>
    public static FloatGrid ReadFromFile(string path, char separator = ',')
    {
        var lines = File.ReadAllLines(path, Encoding.UTF8);
        if (lines.Length == 0)
            return new FloatGrid(0, 0, Array.Empty<double>(), Array.Empty<bool>());

        var rows = new List<double[]>();
        int maxCols = 0;
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = SplitCsvLine(line, separator);
            maxCols = Math.Max(maxCols, parts.Count);
            var row = new double[parts.Count];
            for (int c = 0; c < parts.Count; c++)
                row[c] = ParseCell(parts[c]);
            rows.Add(row);
        }

        if (rows.Count == 0)
            return new FloatGrid(0, 0, Array.Empty<double>(), Array.Empty<bool>());

        int nRows = rows.Count;
        int nCols = maxCols;
        var data = new double[nRows * nCols];
        var orig = new bool[nRows * nCols];
        for (int r = 0; r < nRows; r++)
        {
            var row = rows[r];
            for (int c = 0; c < nCols; c++)
            {
                double v = c < row.Length ? row[c] : double.NaN;
                int i = r * nCols + c;
                data[i] = v;
                orig[i] = FloatGrid.IsFinite(v);
            }
        }

        return new FloatGrid(nRows, nCols, data, orig);
    }

    public static void WriteToFile(FloatGrid grid, string path)
    {
        var sb = new StringBuilder();
        for (int r = 0; r < grid.Rows; r++)
        {
            for (int c = 0; c < grid.Cols; c++)
            {
                if (c > 0) sb.Append(',');
                double v = grid[r, c];
                sb.Append(FloatGrid.IsFinite(v) ? v.ToString(CultureInfo.InvariantCulture) : "");
            }
            sb.AppendLine();
        }
        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
    }

    private static double ParseCell(string s)
    {
        var t = s.Trim();
        if (t.Length == 0) return double.NaN;
        foreach (var m in MissingTokens)
            if (t.Equals(m, StringComparison.OrdinalIgnoreCase)) return double.NaN;
        if (double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) return v;
        return double.NaN;
    }

    private static List<string> SplitCsvLine(string line, char sep)
    {
        var list = new List<string>();
        var cur = new StringBuilder();
        bool inQuotes = false;
        for (int i = 0; i < line.Length; i++)
        {
            char ch = line[i];
            if (ch == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"') { cur.Append('"'); i++; }
                else inQuotes = !inQuotes;
            }
            else if (ch == sep && !inQuotes)
            {
                list.Add(cur.ToString());
                cur.Clear();
            }
            else cur.Append(ch);
        }
        list.Add(cur.ToString());
        return list;
    }
}
