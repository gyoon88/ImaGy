using ImaGy.Models;
using Microsoft.Win32;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
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
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var bitmap = await Task.Run(() =>
                {
                    var bmp = new BitmapImage();
                    bmp.BeginInit();
                    bmp.UriSource = new Uri(openDialog.FileName);
                    bmp.CacheOption = BitmapCacheOption.OnLoad;
                    bmp.EndInit();
                    bmp.Freeze();
                    return bmp;
                });
                stopwatch.Stop();

                return new ImageLoadResult
                {
                    Bitmap = bitmap,
                    FileName = System.IO.Path.GetFileName(openDialog.FileName),
                    Resolution = $"{bitmap.PixelWidth}x{bitmap.PixelHeight}",
                    LoadTime = stopwatch.Elapsed.TotalMilliseconds
                };
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
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(openDialog.FileName);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();
                    bitmap.Freeze();
                    return bitmap;
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to load the template image.\n\nError: {ex.Message}",
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
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
    }
}