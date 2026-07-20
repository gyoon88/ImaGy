using OpenCvSharp;

namespace ImaGy.Grids;

public static class GridAlignmentService
{
    public static (FloatGrid AlignedA, FloatGrid AlignedB) Align(FloatGrid a, FloatGrid b, GridAlignmentOptions opt)
    {
        return opt.Mode switch
        {
            GridAlignmentMode.Crop => AlignCrop(a, b),
            GridAlignmentMode.Pad => AlignPad(a, b, opt.PadPlacement),
            GridAlignmentMode.Resample => AlignResample(a, b, opt.ResampleReference, opt.Interpolation),
            _ => AlignCrop(a, b)
        };
    }

    private static (FloatGrid, FloatGrid) AlignCrop(FloatGrid a, FloatGrid b)
    {
        int rows = Math.Min(a.Rows, b.Rows);
        int cols = Math.Min(a.Cols, b.Cols);
        return (Slice(a, 0, 0, rows, cols), Slice(b, 0, 0, rows, cols));
    }

    private static FloatGrid Slice(FloatGrid g, int r0, int c0, int rows, int cols)
    {
        var d = new double[rows * cols];
        var o = new bool[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int si = (r0 + r) * g.Cols + (c0 + c);
                int di = r * cols + c;
                d[di] = g.Data[si];
                o[di] = g.OriginallyFinite[si];
            }
        }
        return new FloatGrid(rows, cols, d, o);
    }

    private static (FloatGrid, FloatGrid) AlignPad(FloatGrid a, FloatGrid b, PadPlacement placement)
    {
        int rows = Math.Max(a.Rows, b.Rows);
        int cols = Math.Max(a.Cols, b.Cols);
        return (PadTo(a, rows, cols, placement), PadTo(b, rows, cols, placement));
    }

    private static FloatGrid PadTo(FloatGrid g, int rows, int cols, PadPlacement placement)
    {
        if (g.Rows == rows && g.Cols == cols) return g.Clone();
        int offR = placement == PadPlacement.TopLeft ? 0 : Math.Max(0, (rows - g.Rows) / 2);
        int offC = placement == PadPlacement.TopLeft ? 0 : Math.Max(0, (cols - g.Cols) / 2);
        var d = Enumerable.Repeat(double.NaN, rows * cols).ToArray();
        var o = new bool[rows * cols];
        for (int r = 0; r < g.Rows; r++)
        {
            for (int c = 0; c < g.Cols; c++)
            {
                int dr = r + offR;
                int dc = c + offC;
                if (dr < 0 || dr >= rows || dc < 0 || dc >= cols) continue;
                int di = dr * cols + dc;
                d[di] = g[r, c];
                o[di] = g.OriginallyFinite[r * g.Cols + c];
            }
        }
        return new FloatGrid(rows, cols, d, o);
    }

    private static (FloatGrid, FloatGrid) AlignResample(FloatGrid a, FloatGrid b, ResampleShapeReference reference, InterpolationFlags interp)
    {
        int rows = reference == ResampleShapeReference.GridA ? a.Rows : b.Rows;
        int cols = reference == ResampleShapeReference.GridA ? a.Cols : b.Cols;
        return (ResampleGrid(a, rows, cols, interp), ResampleGrid(b, rows, cols, interp));
    }

    private static FloatGrid ResampleGrid(FloatGrid g, int rows, int cols, InterpolationFlags interp)
    {
        if (g.Rows == rows && g.Cols == cols) return g.Clone();
        using var src = ToMat64(g);
        using var dst = new Mat();
        Cv2.Resize(src, dst, new Size(cols, rows), 0, 0, interp);
        using var mSrc = ToMaskU8(g.OriginallyFinite, g.Rows, g.Cols);
        using var mDst = new Mat();
        Cv2.Resize(mSrc, mDst, new Size(cols, rows), 0, 0, InterpolationFlags.Nearest);
        var d = new double[rows * cols];
        var o = new bool[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int i = r * cols + c;
                d[i] = dst.Get<double>(r, c);
                o[i] = mDst.Get<byte>(r, c) >= 128;
            }
        }
        return new FloatGrid(rows, cols, d, o);
    }

    private static Mat ToMat64(FloatGrid g)
    {
        var m = new Mat(g.Rows, g.Cols, MatType.CV_64FC1);
        for (int r = 0; r < g.Rows; r++)
            for (int c = 0; c < g.Cols; c++)
                m.Set(r, c, g[r, c]);
        return m;
    }

    private static Mat ToMaskU8(bool[] mask, int rows, int cols)
    {
        var m = new Mat(rows, cols, MatType.CV_8UC1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m.Set(r, c, mask[r * cols + c] ? (byte)255 : (byte)0);
        return m;
    }
}
