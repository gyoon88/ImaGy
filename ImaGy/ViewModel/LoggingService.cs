using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;

namespace ImaGy.ViewModel
{
    public class LoggingService : BaseViewModel
    {
        private string logText = "";
        public string LogText
        {
            get => logText;
            set => SetProperty(ref logText, value);
        }

        public void AddLog(string message)
        {
            string timestamp = DateTime.Now.ToString("[yyyy-MM-dd HH:mm:ss]");
            LogText += $"{timestamp} [INFO] {message}{Environment.NewLine}";
        }

        public void ExportToCsv(string filePath)
        {
            var headers = new[] { "Timestamp", "Level", "Message" };
            var lines = LogText.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);
            var data = lines.Select(line =>
            {
                // Example line: "[2024-09-04 10:30:00] [INFO] Application started."
                var match = Regex.Match(line, @"\[(.*?)\] \[(.*?)\] (.*)");
                if (match.Success)
                {
                    return new[] { match.Groups[1].Value, match.Groups[2].Value, match.Groups[3].Value.Trim() };
                }
                return new[] { "", "", line }; // Fallback
            });

            CsvExporter.Export(filePath, data, headers);
        }
    }
}