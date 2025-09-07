using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using ImaGy.ViewModels;

namespace ImaGy.Services
{
    public class HistoryService : BaseViewModel
    {
        public ObservableCollection<string> HistoryItems { get; } = new ObservableCollection<string>();

        public void AddHistory(string message, double elapsedMs)
        {
            string timestamp = DateTime.Now.ToString("[HH:mm:ss]");
            string formattedMessage = $"{timestamp} {message} ({elapsedMs:F2} ms)";
            HistoryItems.Insert(0, formattedMessage);
        }

        public void ExportToCsv(string filePath)
        {
            var headers = new[] { "Timestamp", "Message", "ElapsedTime(ms)" };
            var data = HistoryItems.Select(item =>
            {
                // Example item: "[14:32:01] Binarization (C#) (13.45 ms)"
                var match = Regex.Match(item, @"\[(.*?)\] (.*?) \((\d+\.\d{2}) ms\)");
                if (match.Success)
                {
                    return new[] { match.Groups[1].Value, match.Groups[2].Value.Trim(), match.Groups[3].Value };
                }
                return new[] { "", item, "" }; // Fallback for non-matching lines
            });

            CsvExporter.Export(filePath, data, headers);
        }

        public void Clear() => HistoryItems.Clear();
    }

}