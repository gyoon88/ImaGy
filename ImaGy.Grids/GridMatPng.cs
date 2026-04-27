using OpenCvSharp;

namespace ImaGy.Grids;

public static class GridMatPng
{
    public static byte[] EncodePng(Mat mat)
    {
        Cv2.ImEncode(".png", mat, out var buf);
        return buf.ToArray();
    }
}
