using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace ImaGy.ViewModel
{
    internal class BaseViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
