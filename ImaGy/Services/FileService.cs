using ImaGy.Models;
using Microsoft.Win32;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImaGy.Services
{
    public class FileService
    {
        public async Task<ImageLoadResult?> OpenImage()
        {
            OpenFileDialog openDialog = new OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpeg;*.jpg;*.bmp)|*.png;*.jpeg;*.jpg;*.bmp|All files (*.*)|*.*"
            };

            if (openDialog.ShowDialog() == true)
            {
                try
                {
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                    // 새로운 표준화 메서드를 비동기적으로 호출
                    var bitmap = await Task.Run(() => LoadAndStandardizeImage(openDialog.FileName));

                    stopwatch.Stop();

                    return new ImageLoadResult
                    {
                        Bitmap = bitmap,
                        FileName = System.IO.Path.GetFileName(openDialog.FileName),
                        Resolution = $"{bitmap.PixelWidth}x{bitmap.PixelHeight}",
                        LoadTime = stopwatch.Elapsed.TotalMilliseconds
                    };
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지를 여는 데 실패했습니다.\n\n오류: {ex.Message}",
                                    "파일 열기 오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            return null;
        }

        public string? GetSaveFilePath(string defaultFileName)
        {
            SaveFileDialog saveDialog = new SaveFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv",
                FileName = defaultFileName
            };

            if (saveDialog.ShowDialog() == true)
            {
                return saveDialog.FileName;
            }
            return null;
        }

        /// <summary>
        /// REFACTOR: 템플릿 이미지를 열고 표준 포맷으로 변환
        /// </summary>
        public BitmapSource? OpenTemplateImage()
        {
            OpenFileDialog openDialog = new OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpeg;*.jpg;*.bmp)|*.png;*.jpeg;*.jpg;*.bmp|All files (*.*)|*.*"
            };
            if (openDialog.ShowDialog() == true)
            {
                try
                {
                    // REFACTOR: 새로운 표준화 메서드를 직접 호출
                    return LoadAndStandardizeImage(openDialog.FileName);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"템플릿 이미지를 여는 데 실패했습니다.\n\n오류: {ex.Message}",
                                    "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            return null;
        }


        public void SaveImage(BitmapSource image)
        {
            if (image == null) return;

            SaveFileDialog saveDialog = new SaveFileDialog
            {
                Filter = "PNG Image|*.png|JPEG Image|*.jpg|BMP Image|*.bmp"
            };

            if (saveDialog.ShowDialog() == true)
            {
                try
                {
                    BitmapEncoder? encoder = Path.GetExtension(saveDialog.FileName).ToLower() switch
                    {
                        ".png" => new PngBitmapEncoder(),
                        ".jpg" or ".jpeg" => new JpegBitmapEncoder(),
                        ".bmp" => new BmpBitmapEncoder(),
                        _ => null
                    };

                    if (encoder != null)
                    {
                        encoder.Frames.Add(BitmapFrame.Create(image));
                        using var fileStream = new FileStream(saveDialog.FileName, FileMode.Create);
                        encoder.Save(fileStream);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to save the image.\n\nError: {ex.Message}",
                     "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

    /// <summary>
    /// NEW: 지정된 경로의 이미지를 로드하고 Gray8 또는 Bgra32 포맷으로 표준화하는 헬퍼 메서드
    /// </summary>
    /// <param name="filePath">이미지 파일 경로</param>
    /// <returns>표준화된 BitmapSource</returns>
    private BitmapSource LoadAndStandardizeImage(string filePath)
    {
        BitmapImage originalBitmap = new BitmapImage();
        originalBitmap.BeginInit();
        originalBitmap.UriSource = new Uri(filePath);
        originalBitmap.CacheOption = BitmapCacheOption.OnLoad;
        originalBitmap.EndInit();
        originalBitmap.Freeze();

        if (originalBitmap.Format == PixelFormats.Gray8 ||  originalBitmap.Format == PixelFormats.BlackWhite)
        {
            if (originalBitmap.Format == PixelFormats.Gray8)
            {
                return originalBitmap;
            }
            var grayBitmap = new FormatConvertedBitmap(originalBitmap, PixelFormats.Gray8, null, 0);
            grayBitmap.Freeze();
            return grayBitmap;
        }
        else
        {
            if (originalBitmap.Format == PixelFormats.Bgra32)
            {
                return originalBitmap;

            }               
            var colorBitmap = new FormatConvertedBitmap(originalBitmap, PixelFormats.Bgra32, null, 0);
            colorBitmap.Freeze();
            return colorBitmap;
            }
        }
    }
}
