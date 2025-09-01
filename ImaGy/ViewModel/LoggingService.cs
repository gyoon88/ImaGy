using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}
