using ImaGy.Wrapper;
using System.Windows.Media.Imaging;

namespace ImaGy.Models
{
    public class MorphologyProcessor
    {
        // Mophorogy
        public BitmapSource ApplyDilation(BitmapSource source, int kernelSize, bool useCircularKernel, bool isColor)
        {
            if (isColor)
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyDilationColor(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
                });
            }
            else
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyDilation(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
                });
            }


        }

        public BitmapSource ApplyErosion(BitmapSource source, int kernelSize, bool useCircularKernel, bool isColor)
        {
            if (isColor)
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyErosionColor(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
                });
            }
            else
            {
                return BitmapProcessorHelper.ApplyKernelEffect(source, kernelSize, (pixelPtr, width, height, stride) =>
                {
                    NativeProcessor.ApplyErosion(pixelPtr, width, height, stride, kernelSize, useCircularKernel);
                });
            }
        }
    }
}