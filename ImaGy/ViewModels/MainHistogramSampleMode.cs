namespace ImaGy.ViewModels;

/// <summary>메인 이미지 히스토그램 표본 종류.</summary>
public enum MainHistogramSampleMode
{
    /// <summary>픽셀 단위 밝기(또는 RGB 채널).</summary>
    PixelIntensity = 0,
    /// <summary>각 행마다 X방향(가로) 밝기 합의 분포.</summary>
    RowSumAlongX = 1,
    /// <summary>각 열마다 Y방향(세로) 밝기 합의 분포.</summary>
    ColSumAlongY = 2
}
