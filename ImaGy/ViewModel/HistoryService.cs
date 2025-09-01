using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaGy.ViewModel
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

        public void Clear() => HistoryItems.Clear();
    }

}
