namespace ImaGy.Grids;

public static class GridRoiMaskBuilder
{
    public static bool[] BuildMask(int rows, int cols, GridRoiDefinition roi)
    {
        var m = new bool[rows * cols];
        switch (roi.Kind)
        {
            case RoiKind.Rectangle:
                FillRect(m, rows, cols, roi.X, roi.Y, roi.Width, roi.Height);
                break;
            case RoiKind.Triangle:
                FillTriangle(m, rows, cols, roi.C0, roi.R0, roi.C1, roi.R1, roi.C2, roi.R2);
                break;
            case RoiKind.Circle:
                FillCircle(m, rows, cols, roi.CenterCol, roi.CenterRow, roi.RadiusPixels);
                break;
        }
        return m;
    }

    private static void FillRect(bool[] m, int rows, int cols, int x, int y, int w, int h)
    {
        for (int r = y; r < y + h && r < rows; r++)
        {
            if (r < 0) continue;
            for (int c = x; c < x + w && c < cols; c++)
            {
                if (c < 0) continue;
                m[r * cols + c] = true;
            }
        }
    }

    private static void FillTriangle(bool[] m, int rows, int cols, int c0, int r0, int c1, int r1, int c2, int r2)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (PointInTriangle(c, r, c0, r0, c1, r1, c2, r2))
                    m[r * cols + c] = true;
            }
        }
    }

    private static bool PointInTriangle(int px, int py, int x1, int y1, int x2, int y2, int x3, int y3)
    {
        static int orient(int ax, int ay, int bx, int by, int cx, int cy) =>
            (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

        int o1 = orient(x1, y1, x2, y2, px, py);
        int o2 = orient(x2, y2, x3, y3, px, py);
        int o3 = orient(x3, y3, x1, y1, px, py);
        bool hasNeg = o1 < 0 || o2 < 0 || o3 < 0;
        bool hasPos = o1 > 0 || o2 > 0 || o3 > 0;
        return !(hasNeg && hasPos);
    }

    private static void FillCircle(bool[] m, int rows, int cols, int cc, int cr, double radius)
    {
        double r2 = radius * radius;
        int ri = (int)Math.Ceiling(radius);
        for (int r = Math.Max(0, cr - ri); r < Math.Min(rows, cr + ri + 1); r++)
        {
            for (int c = Math.Max(0, cc - ri); c < Math.Min(cols, cc + ri + 1); c++)
            {
                double dx = c - cc;
                double dy = r - cr;
                if (dx * dx + dy * dy <= r2)
                    m[r * cols + c] = true;
            }
        }
    }
}
