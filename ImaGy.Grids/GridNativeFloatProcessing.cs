namespace ImaGy.Grids;

/// <summary>
/// Placeholder for future ImaGyNative float-grid overloads (convolution, IQR SIMD, etc.).
/// Current pipeline runs in managed code + OpenCvSharp resize in alignment.
/// </summary>
public static class GridNativeFloatProcessing
{
    public static bool IsNativeAccelerationAvailable => false;
}
