using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace ImaGy.Model
{
    internal class ImageDocument
    {

        private BitmapImage OpenImage(object parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            // 이미지 파일 형식 필터 설정
            openDialog.Filter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png";
            BitmapImage image = null;
            // 사용자가 파일을 선택하고 '열기' 버튼을 눌렀을 때
            if (openDialog.ShowDialog() == true)
            {
                image =  new BitmapImage(new Uri(openDialog.FileName));
            }
            return image;
        }

        // Method Saving iamge
        private void SaveImage(BitmapImage image)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            // 이미지 파일 형식 필터 설정
            openDialog.Filter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png";

            // 사용자가 파일을 선택하고 '저장' 버튼을 눌렀을 때

        }

        private void SaveImageOtherName(object parameter)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            // 이미지 파일 형식 필터 설정
            openDialog.Filter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png";

            // 사용자가 파일을 선택하고 '저장' 버튼을 눌렀을 때

        }

    }
}
