using OpenCvSharp;

namespace ImaGy.Grids;

internal static class GridMatHelpers
{
    public static Mat ToMat64(FloatGrid g)
    {
        var m = new Mat(g.Rows, g.Cols, MatType.CV_64FC1);
        for (int r = 0; r < g.Rows; r++)
            for (int c = 0; c < g.Cols; c++)
                m.Set(r, c, g[r, c]);
        return m;
    }

    public static Mat ToMaskU8(bool[] mask, int rows, int cols)
    {
        var m = new Mat(rows, cols, MatType.CV_8UC1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m.Set(r, c, mask[r * cols + c] ? (byte)255 : (byte)0);
        return m;
    }
}
