using System;
using System.Windows.Input;

namespace ImaGy.ViewModels.Commands
{
    public class RelayCommand<T> : ICommand
    {
        private readonly Action<T?> _execute;
        private readonly Predicate<T?>? _canExecute;

        public RelayCommand(Action<T?> execute, Predicate<T?>? canExecute = null)
        {
            _execute = execute ?? throw new ArgumentNullException(nameof(execute));
            _canExecute = canExecute;
        }

        public event EventHandler? CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }

        public bool CanExecute(object? parameter)
        {
            // To prevent exceptions when the parameter is of a different type,
            // we check the type before casting.
            if (parameter is T validParameter)
            {
                return _canExecute == null || _canExecute(validParameter);
            }
            // If the parameter is null and T is a reference type, we can proceed.
            if (parameter == null && !typeof(T).IsValueType)
            {   
                // The cast (T?)parameter will be null, which might be acceptable for the predicate.
                return _canExecute == null || _canExecute((T?)parameter);
            }
            // In other cases, the execution is not possible.
            return false;
        }

        public void Execute(object? parameter)
        {
            if (parameter is T validParameter)
            {
                _execute(validParameter);
            }
            else if (parameter == null && !typeof(T).IsValueType)
            {
                _execute((T?)parameter);
            }
        }
    }
    
    // 파라미터가 없는 버전
    public class RelayCommand : ICommand
    {
        private readonly Action _execute;
        private readonly Func<bool>? _canExecute;

        public RelayCommand(Action execute, Func<bool>? canExecute = null)
        {
            _execute = execute;
            _canExecute = canExecute;
        }

        public event EventHandler? CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object? parameter)
        {
            return _canExecute == null || _canExecute();
        }

        public void Execute(object? parameter)
        {
            _execute();
        }
    }
}