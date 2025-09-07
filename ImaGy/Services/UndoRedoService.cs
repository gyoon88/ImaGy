using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ImaGy.ViewModels;

namespace ImaGy.Services
{
    public class UndoRedoService<T> : BaseViewModel
    {
        private readonly Stack<T?> undoStack = new Stack<T?>();
        private readonly Stack<T?> redoStack = new Stack<T?>();

        public bool CanUndo => undoStack.Any();
        public bool CanRedo => redoStack.Any();

        public void AddState(T? state)
        {
            undoStack.Push(state);
            redoStack.Clear();
            // CanUndo, CanRedo 프로퍼티가 변경되었음을 UI에 알림
            OnPropertyChanged(nameof(CanUndo));
            OnPropertyChanged(nameof(CanRedo));
        }

        public T? Undo(T? currentState)
        {
            if (!CanUndo) return default;

            redoStack.Push(currentState);
            T? previousState = undoStack.Pop();

            OnPropertyChanged(nameof(CanUndo));
            OnPropertyChanged(nameof(CanRedo));

            return previousState;
        }

        public T? Redo(T? currentState)
        {
            if (!CanRedo) return default;

            undoStack.Push(currentState);
            T? nextState = redoStack.Pop();

            OnPropertyChanged(nameof(CanUndo));
            OnPropertyChanged(nameof(CanRedo));

            return nextState;
        }

        public void Clear()
        {
            undoStack.Clear();
            redoStack.Clear();
            OnPropertyChanged(nameof(CanUndo));
            OnPropertyChanged(nameof(CanRedo));
        }
    }
}
