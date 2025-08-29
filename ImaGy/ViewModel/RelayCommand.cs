using System;
using System.Windows.Input;

namespace ImaGy.ViewModel
{
    internal class RelayCommand : ICommand
    {
        private readonly Action<object> _execute; // 실행 코드 반환 X 실행할 메서드 전달
        private readonly Predicate<object> _canExecute; // boolian 조건 연산을 줌 

        public RelayCommand(Action<object> execute, Predicate<object> canExecute = null)
        {
            _execute = execute ?? throw new ArgumentNullException(nameof(execute)); // 삼항 연산자
            _canExecute = canExecute; // null 초기화로 항상 실행
        }

        // 조건 연산 
        public bool CanExecute(object parameter) => _canExecute == null || _canExecute(parameter);
        public void Execute(object parameter) => _execute(parameter);
        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }
    }
}
