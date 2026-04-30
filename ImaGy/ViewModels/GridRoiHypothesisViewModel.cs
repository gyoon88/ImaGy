using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Windows.Input;
using ImaGy.Grids;
using ImaGy.ViewModels.Commands;

namespace ImaGy.ViewModels;

public sealed class GridRoiHypothesisViewModel : BaseViewModel
{
    private readonly GridWorkbenchViewModel _workbench;
    private GridRoiDefinition? _roiGroup1;
    private GridRoiDefinition? _roiGroup2;
    private string _resultText = "그룹 ROI를 각각 선택한 뒤 「검정 실행」을 누르세요.";

    public GridRoiHypothesisViewModel(GridWorkbenchViewModel workbench)
    {
        _workbench = workbench;
        RunTestsCommand = new RelayCommand(RunTests, CanRun);
    }

    public ObservableCollection<GridRoiDefinition> RoiChoices => _workbench.DrawnRois;

    public GridRoiDefinition? RoiGroup1
    {
        get => _roiGroup1;
        set
        {
            if (SetProperty(ref _roiGroup1, value))
                CommandManager.InvalidateRequerySuggested();
        }
    }

    public GridRoiDefinition? RoiGroup2
    {
        get => _roiGroup2;
        set
        {
            if (SetProperty(ref _roiGroup2, value))
                CommandManager.InvalidateRequerySuggested();
        }
    }

    public string ResultText
    {
        get => _resultText;
        private set => SetProperty(ref _resultText, value);
    }

    public ICommand RunTestsCommand { get; }

    private bool CanRun() =>
        _workbench.CurrentPipelineResult != null
        && RoiGroup1 != null
        && RoiGroup2 != null
        && !ReferenceEquals(RoiGroup1, RoiGroup2);

    private void RunTests()
    {
        var grid = _workbench.CurrentPipelineResult!.Diff;
        var r1 = RoiGroup1!;
        var r2 = RoiGroup2!;
        var m1 = GridRoiMaskBuilder.BuildMask(grid.Rows, grid.Cols, r1);
        var m2 = GridRoiMaskBuilder.BuildMask(grid.Rows, grid.Cols, r2);
        var s1 = GridRoiHypothesisTests.ExtractFiniteSample(grid, m1);
        var s2 = GridRoiHypothesisTests.ExtractFiniteSample(grid, m2);
        if (s1.Count < 2 || s2.Count < 2)
        {
            ResultText = "각 ROI 안에 유한한 Diff 값이 최소 2개씩 있어야 합니다.";
            return;
        }

        var sb = new StringBuilder();
        sb.AppendLine($"Diff: {grid.Rows}×{grid.Cols}");
        sb.AppendLine($"그룹 1: «{r1.Name}»  n={s1.Count}");
        sb.AppendLine($"그룹 2: «{r2.Name}»  n={s2.Count}");
        sb.AppendLine($"평균₁={s1.Average():G9}  평균₂={s2.Average():G9}");
        sb.AppendLine();

        var tests = GridRoiHypothesisTests.RunAll(s1, s2);
        if (tests.Count == 0)
        {
            ResultText = sb + "검정을 계산할 수 없습니다.";
            return;
        }

        foreach (var t in tests)
        {
            sb.AppendLine($"── {t.Name} ──");
            sb.AppendLine($"  통계량: {t.Statistic:G9}");
            if (t.DfBetween is { } dfb && t.DfWithin is { } dfw)
                sb.AppendLine($"  df: {dfb.ToString(CultureInfo.InvariantCulture)}, {dfw.ToString(CultureInfo.InvariantCulture)}");
            else if (t.DfWithin is { } dfOnly)
                sb.AppendLine($"  df: {dfOnly.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"  p: {FormatP(t.PValueTwoSided)}");
            sb.AppendLine($"  {t.Note}");
            sb.AppendLine();
        }

        sb.AppendLine("※ 해석은 데이터 가정(정규성·동질성·독립)을 확인한 뒤 진행하세요. Mann–Whitney는 정규 근사입니다.");
        ResultText = sb.ToString();
    }

    private static string FormatP(double p)
    {
        if (double.IsNaN(p)) return "—";
        if (p < 1e-6) return "<1e-6";
        return p.ToString("G6", CultureInfo.InvariantCulture);
    }
}
