
using System.Diagnostics;
namespace ImaGy.Model
{
    public class ProcessTime
    {
        public static (T result, double elapsedMilliseconds) Measure<T>(Func<T> func)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            T result = func(); // 전달받은 함수 실행 및 결과 저장

            stopwatch.Stop();
            return (result, stopwatch.Elapsed.TotalMilliseconds);
        }

        // 반환 값이 없는 함수(Action)의 실행 시간을 측정합니다.
        public static double Measure(Action action)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            action(); // 전달받은 함수 실행

            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds;
        }
    }
}
