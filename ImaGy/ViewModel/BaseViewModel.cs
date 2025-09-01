using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace ImaGy.ViewModel
{
    public class BaseViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        // 바로 이 부분이 빠져있는 헬퍼 메서드입니다.
        protected bool SetProperty<T>(ref T field, T newValue, [CallerMemberName] string? propertyName = null)
        {
            // 기존 값과 새 값이 같으면 아무것도 하지 않음 (불필요한 UI 갱신 방지)
            if (EqualityComparer<T>.Default.Equals(field, newValue))
            {
                return false;
            }

            // 새 값으로 필드 업데이트
            field = newValue;

            // View에 프로퍼티가 변경되었음을 알림
            OnPropertyChanged(propertyName);

            return true;
        }
    }
}
