using ImaGy.Services;
using System;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace ImaGy.ViewModels.Commands
{
    public class UndoCommand : ICommand
    {
        private readonly MainViewModel _mainViewModel;
        private readonly UndoRedoService<BitmapSource?> _undoRedoService;

        public UndoCommand(MainViewModel mainViewModel, UndoRedoService<BitmapSource?> undoRedoService)
        {
            _mainViewModel = mainViewModel;
            _undoRedoService = undoRedoService;
        }

        public event EventHandler? CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object? parameter)
        {
            return _undoRedoService.CanUndo;
        }

        public void Execute(object? parameter)
        {
            _mainViewModel.AfterImage = _undoRedoService.Undo(_mainViewModel.AfterImage);
        }
    }
}