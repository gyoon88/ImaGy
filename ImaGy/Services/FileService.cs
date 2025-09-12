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

                    // ���ο� ǥ��ȭ �޼��带 �񵿱������� ȣ��
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
                    MessageBox.Show($"�̹����� ���� �� �����߽��ϴ�.\n\n����: {ex.Message}",
                                    "���� ���� ����", MessageBoxButton.OK, MessageBoxImage.Error);
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
        /// REFACTOR: ���ø� �̹����� ���� ǥ�� �������� ��ȯ
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
                    // REFACTOR: ���ο� ǥ��ȭ �޼��带 ���� ȣ��
                    return LoadAndStandardizeImage(openDialog.FileName);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"���ø� �̹����� ���� �� �����߽��ϴ�.\n\n����: {ex.Message}",
                                    "����", MessageBoxButton.OK, MessageBoxImage.Error);
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
    /// NEW: ������ ����� �̹����� �ε��ϰ� Gray8 �Ǵ� Bgra32 �������� ǥ��ȭ�ϴ� ���� �޼���
    /// </summary>
    /// <param name="filePath">�̹��� ���� ���</param>
    /// <returns>ǥ��ȭ�� BitmapSource</returns>
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
