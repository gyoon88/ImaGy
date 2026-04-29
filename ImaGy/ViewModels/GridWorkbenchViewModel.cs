using ImaGy.Grids;
using ImaGy.Services;
using ImaGy.View;
using ImaGy.ViewModels.Commands;
using OpenCvSharp;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace ImaGy.ViewModels;

public sealed class GridWorkbenchViewModel : BaseViewModel
{
    private readonly LoggingService _log;
    private readonly SynchronizationContext? _uiSync;
    private string _pathA = "";
    private string _pathB = "";
    private string _pathDiff = "";
    private string _outputFolder = "";
    private BitmapSource? _previewA;
    private BitmapSource? _previewB;
    private BitmapSource? _previewDiff;
    private GridPipelineResult? _lastResult;
    private string _alignmentMode = "Crop";
    private string _padPlacement = "TopLeft";
    private string _resampleRef = "GridA";
    private string _interpolation = "Linear";
    private string _preOrder = "FillThenFilter";
    private string _fillMethod = "None";
    private string _outlierMethod = "None";
    private string _convWindowText = "3";
    private string _fillIterationsText = "2";
    private double _iqrK = 1.5;
    private double _sigmaN = 3;
    private string _combineOp = "Subtract";
    private string _maskRule = "BothValid";
    private string _diffPresentation = "Signed";
    private string _clipMinText = "";
    private string _clipMaxText = "";
    private bool _symmetricSigned = true;
    private string _cmapInput = "Turbo";
    private string _cmapDiff = "Turbo";
    private string _diffDisplayMinText = "";
    private string _diffDisplayMaxText = "";
    private string _diffPreviewMode = "히트맵";
    private string _roiJsonPath = "";
    private string _roiDrawTool = "없음";
    private readonly ObservableCollection<GridRoiDefinition> _drawnRois = new();
    private GridRoiDefinition? _selectedRoi;
    private string _newRoiName = "roi001_rectangle";
    private string _batchFolder = "";
    private string _catalogPath = "";
    private string _summarizeOutCsv = "";
    private string _diffCursorInfo = "x=-, y=-, value=-, norm=-";
    private string _selectedRoiStatsText = "ROI 통계 없음";
    private double _diffDisplayMin;
    private double _diffDisplayMax = 1;
    private bool _isGridBusy;
    private string _gridBusyStatus = "";
    private int _diffPreviewToken;
    private int _gridBusyDepth;

    public GridWorkbenchViewModel(LoggingService log)
    {
        _log = log;
        _uiSync = SynchronizationContext.Current;
        OpenCsvACommand = new RelayCommand(OpenCsvA, () => !IsGridBusy);
        OpenCsvBCommand = new RelayCommand(OpenCsvB, () => !IsGridBusy);
        OpenDiffCsvCommand = new RelayCommand(() => _ = OpenDiffCsvAsync(), () => !IsGridBusy);
        RunPipelineCommand = new RelayCommand(() => _ = RunPipelineAsync(), () => File.Exists(PathA) && File.Exists(PathB) && !IsGridBusy);
        SaveDiffCsvCommand = new RelayCommand(SaveDiffCsv, () => _lastResult != null && !IsGridBusy);
        SaveHeatmapsCommand = new RelayCommand(() => _ = SaveHeatmapsAsync(), () => _lastResult != null && !IsGridBusy);
        SaveScottPlotCommand = new RelayCommand(() => _ = SaveScottPlotAsync(), () => _lastResult != null && !IsGridBusy);
        OpenHistogramCommand = new RelayCommand(OpenHistogram, () => _lastResult != null);
        OpenLineProfileCommand = new RelayCommand(OpenLineProfile, () => _lastResult != null);
        PickOutputFolderCommand = new RelayCommand(PickOutputFolder, () => !IsGridBusy);
        LoadRoiJsonCommand = new RelayCommand(LoadRoiJson, () => !IsGridBusy);
        RunBatchFolderCommand = new RelayCommand(() => _ = RunBatchFolderAsync(), () => Directory.Exists(BatchFolder) && !IsGridBusy);
        SummarizeDiffsCommand = new RelayCommand(() => _ = SummarizeDiffsAsync(), () => Directory.Exists(BatchFolder) && (File.Exists(CatalogPath) || _drawnRois.Count > 0) && !IsGridBusy);
        ApplyDiffPreviewCommand = new RelayCommand(() => _ = ScheduleRefreshDiffPreviewAsync(), () => _lastResult != null && !IsGridBusy);
        PickBatchFolderCommand = new RelayCommand(PickBatchFolder);
        PickBatchCatalogJsonCommand = new RelayCommand(PickBatchCatalogJson);
        PickSummarizeOutCsvCommand = new RelayCommand(PickSummarizeOutCsv);
    }

    public string PathA { get => _pathA; set => SetProperty(ref _pathA, value); }
    public string PathB { get => _pathB; set => SetProperty(ref _pathB, value); }
    public string PathDiff { get => _pathDiff; set => SetProperty(ref _pathDiff, value); }
    public string OutputFolder { get => _outputFolder; set => SetProperty(ref _outputFolder, value); }
    public BitmapSource? PreviewA { get => _previewA; private set => SetProperty(ref _previewA, value); }
    public BitmapSource? PreviewB { get => _previewB; private set => SetProperty(ref _previewB, value); }
    public BitmapSource? PreviewDiff { get => _previewDiff; private set => SetProperty(ref _previewDiff, value); }
    public string AlignmentMode { get => _alignmentMode; set => SetProperty(ref _alignmentMode, value); }
    public string PadPlacement { get => _padPlacement; set => SetProperty(ref _padPlacement, value); }
    public string ResampleRef { get => _resampleRef; set => SetProperty(ref _resampleRef, value); }
    public string Interpolation { get => _interpolation; set => SetProperty(ref _interpolation, value); }
    public string PreOrder { get => _preOrder; set => SetProperty(ref _preOrder, value); }
    public string FillMethod { get => _fillMethod; set => SetProperty(ref _fillMethod, value); }
    public string OutlierMethod { get => _outlierMethod; set => SetProperty(ref _outlierMethod, value); }
    public string ConvWindowText { get => _convWindowText; set => SetProperty(ref _convWindowText, value); }
    public string FillIterationsText { get => _fillIterationsText; set => SetProperty(ref _fillIterationsText, value); }
    public double IqrK { get => _iqrK; set => SetProperty(ref _iqrK, value); }
    public double SigmaN { get => _sigmaN; set => SetProperty(ref _sigmaN, value); }
    public string CombineOp { get => _combineOp; set => SetProperty(ref _combineOp, value); }
    public string MaskRule { get => _maskRule; set => SetProperty(ref _maskRule, value); }
    public string DiffPresentation { get => _diffPresentation; set => SetProperty(ref _diffPresentation, value); }
    public string ClipMinText { get => _clipMinText; set => SetProperty(ref _clipMinText, value); }
    public string ClipMaxText { get => _clipMaxText; set => SetProperty(ref _clipMaxText, value); }
    public bool SymmetricSigned { get => _symmetricSigned; set => SetProperty(ref _symmetricSigned, value); }
    public string CmapInput { get => _cmapInput; set => SetProperty(ref _cmapInput, value); }
    public string CmapDiff
    {
        get => _cmapDiff;
        set
        {
            if (!SetProperty(ref _cmapDiff, value)) return;
            if (_lastResult != null && DiffPreviewMode == "히트맵")
                RefreshDiffPreview();
        }
    }
    /// <summary>Diff 미리보기 표시 범위(빈칸이면 자동). 둘 다 숫자일 때만 적용.</summary>
    public string DiffDisplayMinText { get => _diffDisplayMinText; set => SetProperty(ref _diffDisplayMinText, value); }
    public string DiffDisplayMaxText { get => _diffDisplayMaxText; set => SetProperty(ref _diffDisplayMaxText, value); }
    /// <summary>히트맵 / 그레이스케일</summary>
    public string DiffPreviewMode
    {
        get => _diffPreviewMode;
        set
        {
            if (!SetProperty(ref _diffPreviewMode, value)) return;
            if (_lastResult != null)
                RefreshDiffPreview();
        }
    }
    public string RoiJsonPath { get => _roiJsonPath; set => SetProperty(ref _roiJsonPath, value); }
    /// <summary>Diff 미리보기 ROI 도구: 없음 / 사각형 / 원 / 삼각형</summary>
    public string RoiDrawTool { get => _roiDrawTool; set => SetProperty(ref _roiDrawTool, value); }
    public ObservableCollection<GridRoiDefinition> DrawnRois => _drawnRois;
    public GridRoiDefinition? SelectedRoi
    {
        get => _selectedRoi;
        set
        {
            if (SetProperty(ref _selectedRoi, value))
            {
                if (value != null)
                    NewRoiName = value.Name;
                OnPropertyChanged(nameof(DrawnRoi));
                UpdateSelectedRoiStats();
            }
        }
    }
    public string NewRoiName { get => _newRoiName; set => SetProperty(ref _newRoiName, value); }
    /// <summary>기존 코드 호환용: 현재 선택 ROI.</summary>
    public GridRoiDefinition? DrawnRoi
    {
        get => SelectedRoi;
        set
        {
            SelectedRoi = value;
            if (value != null && !_drawnRois.Contains(value))
                _drawnRois.Add(value);
        }
    }
    public string BatchFolder
    {
        get => _batchFolder;
        set
        {
            if (!SetProperty(ref _batchFolder, value))
                return;
            CommandManager.InvalidateRequerySuggested();
        }
    }
    public string CatalogPath
    {
        get => _catalogPath;
        set
        {
            if (!SetProperty(ref _catalogPath, value))
                return;
            CommandManager.InvalidateRequerySuggested();
        }
    }
    public string SummarizeOutCsv
    {
        get => _summarizeOutCsv;
        set
        {
            if (!SetProperty(ref _summarizeOutCsv, value))
                return;
            CommandManager.InvalidateRequerySuggested();
        }
    }
    public string DiffCursorInfo { get => _diffCursorInfo; set => SetProperty(ref _diffCursorInfo, value); }
    public string SelectedRoiStatsText { get => _selectedRoiStatsText; private set => SetProperty(ref _selectedRoiStatsText, value); }

    public bool IsGridBusy
    {
        get => _isGridBusy;
        private set
        {
            if (!SetProperty(ref _isGridBusy, value))
                return;
            CommandManager.InvalidateRequerySuggested();
        }
    }

    public string GridBusyStatus
    {
        get => _gridBusyStatus;
        private set => SetProperty(ref _gridBusyStatus, value);
    }

    private void BeginGridBusy(string status)
    {
        var d = Interlocked.Increment(ref _gridBusyDepth);
        if (d == 1)
        {
            GridBusyStatus = status;
            IsGridBusy = true;
        }
    }

    private void EndGridBusy()
    {
        var d = Interlocked.Decrement(ref _gridBusyDepth);
        if (d <= 0)
        {
            Interlocked.Exchange(ref _gridBusyDepth, 0);
            GridBusyStatus = "";
            IsGridBusy = false;
        }
    }

    public ICommand OpenCsvACommand { get; }
    public ICommand OpenCsvBCommand { get; }
    public ICommand OpenDiffCsvCommand { get; }
    public ICommand RunPipelineCommand { get; }
    public ICommand SaveDiffCsvCommand { get; }
    public ICommand SaveHeatmapsCommand { get; }
    public ICommand SaveScottPlotCommand { get; }
    public ICommand OpenHistogramCommand { get; }
    public ICommand OpenLineProfileCommand { get; }
    public ICommand PickOutputFolderCommand { get; }
    public ICommand LoadRoiJsonCommand { get; }
    public ICommand RunBatchFolderCommand { get; }
    public ICommand SummarizeDiffsCommand { get; }
    public ICommand ApplyDiffPreviewCommand { get; }
    public ICommand PickBatchFolderCommand { get; }
    public ICommand PickBatchCatalogJsonCommand { get; }
    public ICommand PickSummarizeOutCsvCommand { get; }

    private void OpenCsvA()
    {
        var d = new Microsoft.Win32.OpenFileDialog { Filter = "CSV|*.csv|All|*.*" };
        if (d.ShowDialog() == true) PathA = d.FileName;
    }

    private void OpenCsvB()
    {
        var d = new Microsoft.Win32.OpenFileDialog { Filter = "CSV|*.csv|All|*.*" };
        if (d.ShowDialog() == true) PathB = d.FileName;
    }

    private void PostLog(string message)
    {
        if (_uiSync != null)
            _uiSync.Post(_ => _log.AddLog(message), null);
        else
            _log.AddLog(message);
    }

    private async Task OpenDiffCsvAsync()
    {
        var d = new Microsoft.Win32.OpenFileDialog { Filter = "CSV|*.csv|All|*.*" };
        if (d.ShowDialog() != true) return;
        PathDiff = d.FileName;
        var path = PathDiff;
        BeginGridBusy("Diff CSV 로드…");
        try
        {
            var (diff, err) = await Task.Run(() =>
            {
                try
                {
                    return (CsvGridReader.ReadFromFile(path), (Exception?)null);
                }
                catch (Exception ex)
                {
                    return (null, ex);
                }
            }).ConfigureAwait(true);

            if (err != null || diff == null)
            {
                _log.AddLog($"Diff CSV 로드 실패: {err?.Message}");
                System.Windows.MessageBox.Show(err?.Message ?? "알 수 없는 오류", "Diff CSV 열기", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Warning);
                return;
            }

            _lastResult = new GridPipelineResult
            {
                AlignedA = diff,
                AlignedB = diff,
                ProcessedA = diff,
                ProcessedB = diff,
                Diff = diff
            };
            PreviewA = null;
            PreviewB = null;

            await RefreshDiffPreviewWorkerAsync(Interlocked.Increment(ref _diffPreviewToken), manageBusy: false).ConfigureAwait(true);
            if (_drawnRois.Count == 0)
            {
                SelectedRoi = null;
                SelectedRoiStatsText = "ROI 통계 없음";
            }
            else
            {
                if (SelectedRoi == null || !_drawnRois.Contains(SelectedRoi))
                    SelectedRoi = _drawnRois.FirstOrDefault();
                UpdateSelectedRoiStats();
            }
            DiffCursorInfo = "x=-, y=-, value=-, norm=-";
            _log.AddLog($"Diff CSV 로드: {path} ({diff.Rows}x{diff.Cols})");
            CommandManager.InvalidateRequerySuggested();
        }
        finally
        {
            EndGridBusy();
        }
    }

    private void PickOutputFolder()
    {
        using var dlg = new System.Windows.Forms.FolderBrowserDialog();
        if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            OutputFolder = dlg.SelectedPath;
    }

    private void PickBatchFolder()
    {
        using var dlg = new System.Windows.Forms.FolderBrowserDialog
        {
            Description = "배치 처리할 상위 폴더 (Distance1/2 페어가 있는 폴더)",
            UseDescriptionForTitle = true
        };
        if (!string.IsNullOrWhiteSpace(BatchFolder) && Directory.Exists(BatchFolder))
            dlg.SelectedPath = BatchFolder;
        if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
        {
            BatchFolder = dlg.SelectedPath;
            CommandManager.InvalidateRequerySuggested();
        }
    }

    private void PickBatchCatalogJson()
    {
        var d = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "ROI 카탈로그 JSON|*.json|All|*.*",
            Title = "배치용 ROI 카탈로그 JSON 선택"
        };
        if (!string.IsNullOrWhiteSpace(CatalogPath))
        {
            try
            {
                d.InitialDirectory = Path.GetDirectoryName(CatalogPath);
                d.FileName = Path.GetFileName(CatalogPath);
            }
            catch { /* ignore */ }
        }
        if (d.ShowDialog() == true)
        {
            CatalogPath = d.FileName;
            CommandManager.InvalidateRequerySuggested();
        }
    }

    private void PickSummarizeOutCsv()
    {
        var d = new Microsoft.Win32.SaveFileDialog
        {
            Filter = "CSV|*.csv|All|*.*",
            Title = "요약 통계를 저장할 CSV 경로",
            FileName = string.IsNullOrWhiteSpace(SummarizeOutCsv) ? "diff_roi_summary.csv" : Path.GetFileName(SummarizeOutCsv),
            DefaultExt = ".csv"
        };
        if (!string.IsNullOrWhiteSpace(SummarizeOutCsv))
        {
            try
            {
                string? dir = Path.GetDirectoryName(Path.GetFullPath(SummarizeOutCsv));
                if (!string.IsNullOrEmpty(dir) && Directory.Exists(dir))
                    d.InitialDirectory = dir;
            }
            catch { /* ignore */ }
        }
        else if (!string.IsNullOrWhiteSpace(BatchFolder) && Directory.Exists(BatchFolder))
            d.InitialDirectory = BatchFolder;

        if (d.ShowDialog() == true)
        {
            SummarizeOutCsv = d.FileName;
            CommandManager.InvalidateRequerySuggested();
        }
    }

    private GridAlignmentOptions BuildAlign() => new()
    {
        Mode = Enum.Parse<GridAlignmentMode>(AlignmentMode, true),
        PadPlacement = Enum.Parse<PadPlacement>(PadPlacement, true),
        ResampleReference = ResampleRef.Contains('B', StringComparison.OrdinalIgnoreCase)
            ? ResampleShapeReference.GridB
            : ResampleShapeReference.GridA,
        Interpolation = Interpolation.Equals("Cubic", StringComparison.OrdinalIgnoreCase)
            ? InterpolationFlags.Cubic
            : InterpolationFlags.Linear
    };

    private GridPreprocessOptions BuildPre() => new()
    {
        Order = PreOrder.StartsWith("Filter", StringComparison.OrdinalIgnoreCase)
            ? PreprocessOrder.FilterThenFill
            : PreprocessOrder.FillThenFilter,
        FillMethod = Enum.Parse<FillMethod>(FillMethod, true),
        OutlierMethod = Enum.Parse<OutlierMethod>(OutlierMethod, true),
        ConvolutionWindowSize = int.TryParse(ConvWindowText, NumberStyles.Integer, CultureInfo.CurrentCulture, out var cw) ? Math.Max(1, cw) : 3,
        FillIterations = int.TryParse(FillIterationsText, NumberStyles.Integer, CultureInfo.CurrentCulture, out var fillIt) ? Math.Max(0, fillIt) : 2,
        IqrK = IqrK,
        SigmaMultiplier = SigmaN
    };

    private GridCombineOptions BuildCombine()
    {
        double? cmin = double.TryParse(ClipMinText, out var a) ? a : null;
        double? cmax = double.TryParse(ClipMaxText, out var b) ? b : null;
        return new GridCombineOptions
        {
            Operation = CombineOp.Equals("Add", StringComparison.OrdinalIgnoreCase) ? CombineOperation.Add : CombineOperation.Subtract,
            MaskRule = Enum.Parse<ValidMaskRule>(MaskRule, true),
            Presentation = Enum.Parse<DiffPresentation>(DiffPresentation, true),
            ClipMin = cmin,
            ClipMax = cmax,
            SymmetricVRangeForSigned = SymmetricSigned
        };
    }

    private async Task RunPipelineAsync()
    {
        if (!File.Exists(PathA) || !File.Exists(PathB))
            return;
        var pathA = PathA;
        var pathB = PathB;
        var align = BuildAlign();
        var pre = BuildPre();
        var combine = BuildCombine();
        BeginGridBusy("파이프라인 실행…");
        try
        {
            var work = await Task.Run(() =>
            {
                try
                {
                    var rawA = CsvGridReader.ReadFromFile(pathA);
                    var rawB = CsvGridReader.ReadFromFile(pathB);
                    var result = GridPipeline.Run(rawA, rawB, align, pre, pre, combine);
                    var vis = new GridVisualizationOptions();
                    var (aLo, aHi) = GridVisualizationService.GetNormalizeRange(result.ProcessedA, vis);
                    var (bLo, bHi) = GridVisualizationService.GetNormalizeRange(result.ProcessedB, vis);
                    byte[] pngA;
                    byte[] pngB;
                    using (var ma = GridVisualizationService.ToGray8Preview(result.ProcessedA, vis, aLo, aHi))
                        pngA = GridMatPng.EncodePng(ma);
                    using (var mb = GridVisualizationService.ToGray8Preview(result.ProcessedB, vis, bLo, bHi))
                        pngB = GridMatPng.EncodePng(mb);
                    return (result, pngA, pngB, (Exception?)null);
                }
                catch (Exception ex)
                {
                    return ((GridPipelineResult?)null, (byte[]?)null, (byte[]?)null, ex);
                }
            }).ConfigureAwait(true);

            var (pipeResult, pngA, pngB, pipeErr) = work;
            if (pipeErr != null)
            {
                _log.AddLog($"Grid pipeline error: {pipeErr.Message}");
                System.Windows.MessageBox.Show(pipeErr.Message, "격자 처리", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
                return;
            }

            _lastResult = pipeResult!;
            _drawnRois.Clear();
            SelectedRoi = null;
            CommandManager.InvalidateRequerySuggested();
            _log.AddLog($"Grid pipeline OK: diff {_lastResult.Diff.Rows}x{_lastResult.Diff.Cols}");
            PreviewA = MatBitmapConverter.FromPngBytes(pngA!);
            PreviewB = MatBitmapConverter.FromPngBytes(pngB!);
            await RefreshDiffPreviewWorkerAsync(Interlocked.Increment(ref _diffPreviewToken), manageBusy: false).ConfigureAwait(true);
            SelectedRoiStatsText = "ROI 통계 없음";
            DiffCursorInfo = "x=-, y=-, value=-, norm=-";
        }
        finally
        {
            EndGridBusy();
        }
    }

    private void SaveDiffCsv()
    {
        if (_lastResult == null) return;
        var dlg = new Microsoft.Win32.SaveFileDialog { Filter = "CSV|*.csv", FileName = "diff.csv" };
        if (dlg.ShowDialog() == true)
        {
            CsvGridReader.WriteToFile(_lastResult.Diff, dlg.FileName);
            _log.AddLog($"Saved diff CSV {dlg.FileName}");
        }
    }

    private async Task SaveHeatmapsAsync()
    {
        if (_lastResult == null) return;
        string dir = OutputFolder;
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
        {
            PickOutputFolder();
            dir = OutputFolder;
        }
        if (!Directory.Exists(dir)) return;
        var result = _lastResult;
        var pathA = PathA;
        var pathB = PathB;
        var cmapIn = CmapInput;
        var cmapDf = CmapDiff;
        var combine = BuildCombine();
        var vis = new GridVisualizationOptions();
        BeginGridBusy("히트맵 PNG 저장…");
        try
        {
            await Task.Run(() =>
            {
                var (aLo, aHi) = GridVisualizationService.GetNormalizeRange(result.ProcessedA, vis);
                var (bLo, bHi) = GridVisualizationService.GetNormalizeRange(result.ProcessedB, vis);
                var (dmin, dmax) = GridCombineService.GetDisplayRange(result.Diff, combine);
                string baseName = Path.GetFileNameWithoutExtension(pathA) + "__" + Path.GetFileNameWithoutExtension(pathB);
                GridVisualizationService.SaveHeatmapPng(result.ProcessedA, Path.Combine(dir, $"{baseName}_A_cmap-{cmapIn}.png"), aLo, aHi, cmapIn, vis);
                GridVisualizationService.SaveHeatmapPng(result.ProcessedB, Path.Combine(dir, $"{baseName}_B_cmap-{cmapIn}.png"), bLo, bHi, cmapIn, vis);
                GridVisualizationService.SaveHeatmapPng(result.Diff, Path.Combine(dir, $"{baseName}_Diff_cmap-{cmapDf}.png"), dmin, dmax, cmapDf, vis);
            }).ConfigureAwait(true);
            _log.AddLog($"Saved heatmaps to {dir}");
        }
        finally
        {
            EndGridBusy();
        }
    }

    private async Task SaveScottPlotAsync()
    {
        if (_lastResult == null) return;
        string dir = OutputFolder;
        if (!Directory.Exists(dir))
        {
            PickOutputFolder();
            dir = OutputFolder;
        }
        if (!Directory.Exists(dir)) return;
        var result = _lastResult;
        var pathA = PathA;
        var pathB = PathB;
        var cmapDf = CmapDiff;
        var combine = BuildCombine();
        BeginGridBusy("ScottPlot PNG 저장…");
        try
        {
            await Task.Run(() =>
            {
                var (dmin, dmax) = GridCombineService.GetDisplayRange(result.Diff, combine);
                string baseName = Path.GetFileNameWithoutExtension(pathA) + "__" + Path.GetFileNameWithoutExtension(pathB);
                string path = Path.Combine(dir, $"{baseName}_Diff_scottplot_cmap-{cmapDf}.png");
                GridScottPlotExporter.SaveMatplotlibStyleHeatmap(path, result.Diff.ToDouble2D(), dmin, dmax, "Diff", cmapDf, 1200, 900);
            }).ConfigureAwait(true);
            string baseName = Path.GetFileNameWithoutExtension(pathA) + "__" + Path.GetFileNameWithoutExtension(pathB);
            _log.AddLog($"Saved ScottPlot heatmap {Path.Combine(dir, $"{baseName}_Diff_scottplot_cmap-{cmapDf}.png")}");
        }
        finally
        {
            EndGridBusy();
        }
    }

    private void OpenHistogram()
    {
        if (_lastResult == null) return;
        bool[]? mask = BuildRoiMaskForHistogram();
        var w = new HistogramWindow
        {
            DataContext = new HistogramViewModel(_lastResult.Diff, mask),
            Title = "히스토그램 (Diff / CSV)"
        };
        w.Show();
    }

    private void OpenLineProfile()
    {
        if (_lastResult == null) return;
        bool[]? mask = BuildRoiMaskForHistogram();
        var w = new LineProfileWindow
        {
            DataContext = new LineProfileViewModel(_lastResult.Diff, mask),
            Title = "라인 프로파일 (Diff / CSV)"
        };
        w.Show();
    }

    /// <summary>그린 ROI 우선, 없으면 ROI JSON 파일.</summary>
    public bool[]? BuildRoiMaskForHistogram()
    {
        if (_lastResult == null) return null;
        if (SelectedRoi != null)
            return GridRoiMaskBuilder.BuildMask(_lastResult.Diff.Rows, _lastResult.Diff.Cols, SelectedRoi);
        if (_drawnRois.Count > 0)
            return BuildUnionMaskFromDrawnRois(_lastResult.Diff.Rows, _lastResult.Diff.Cols);
        if (!File.Exists(RoiJsonPath)) return null;
        return BuildMaskFromRoiFile(_lastResult.Diff);
    }

    public void SaveDrawnRoiToJson(string path)
    {
        if (_lastResult == null || _drawnRois.Count == 0)
            throw new InvalidOperationException("저장할 ROI가 없습니다. Diff 미리보기에서 도구로 그리세요.");

        for (int i = 0; i < _drawnRois.Count; i++)
            NormalizeRoiIdentity(_drawnRois[i], i + 1);

        var catalog = new GridRoiCatalog
        {
            Schema = "imagy.roi.catalog.v1",
            ReferenceRows = _lastResult.Diff.Rows,
            ReferenceCols = _lastResult.Diff.Cols,
            ReferencePairIndex = 0,
            Rois = _drawnRois.Select(ToCatalogEntry).ToList()
        };
        catalog.Save(path);

        RoiJsonPath = path;
        _log.AddLog($"ROI JSON 저장: {path} ({_drawnRois.Count}개)");
    }

    private void LoadRoiJson()
    {
        var d = new Microsoft.Win32.OpenFileDialog { Filter = "JSON|*.json" };
        if (d.ShowDialog() != true) return;
        RoiJsonPath = d.FileName;
        try
        {
            var loaded = _lastResult == null
                ? LoadRoisFromFileWithoutGrid(RoiJsonPath)
                : LoadRoisFromFile(RoiJsonPath, _lastResult.Diff);
            _drawnRois.Clear();
            for (int i = 0; i < loaded.Count; i++)
            {
                var roi = loaded[i];
                NormalizeRoiIdentity(roi, i + 1);
                _drawnRois.Add(roi);
            }
            SelectedRoi = _drawnRois.FirstOrDefault();
            UpdateSelectedRoiStats();
            CommandManager.InvalidateRequerySuggested();
        }
        catch (Exception ex)
        {
            _drawnRois.Clear();
            SelectedRoi = null;
            _log.AddLog($"ROI JSON 로드 실패: {ex.Message}");
            CommandManager.InvalidateRequerySuggested();
        }
    }

    public void UpsertCurrentRoi(GridRoiDefinition roi)
    {
        int selectedIndex = SelectedRoi != null ? Math.Max(0, _drawnRois.IndexOf(SelectedRoi)) : _drawnRois.Count;
        string defaultLabel = BuildDefaultRoiLabel(selectedIndex + 1, roi.Kind);
        string name = string.IsNullOrWhiteSpace(NewRoiName) ? defaultLabel : NewRoiName.Trim();
        var normalized = CloneRoi(roi);
        normalized.Name = name;
        if (string.IsNullOrWhiteSpace(normalized.Id))
            normalized.Id = defaultLabel;

        if (SelectedRoi != null)
        {
            int idx = _drawnRois.IndexOf(SelectedRoi);
            if (idx >= 0)
            {
                _drawnRois[idx] = normalized;
                SelectedRoi = normalized;
                UpdateSelectedRoiStats();
                _log.AddLog($"ROI 갱신: {normalized.Name} ({normalized.Kind})");
                return;
            }
        }

        _drawnRois.Add(normalized);
        SelectedRoi = normalized;
        NewRoiName = BuildDefaultRoiLabel(_drawnRois.Count + 1, roi.Kind);
        UpdateSelectedRoiStats();
        CommandManager.InvalidateRequerySuggested();
        _log.AddLog($"ROI 추가: {normalized.Name} ({normalized.Kind})");
    }

    public void AddDrawnRoi(GridRoiDefinition roi)
    {
        string defaultLabel = BuildDefaultRoiLabel(_drawnRois.Count + 1, roi.Kind);
        string name = string.IsNullOrWhiteSpace(NewRoiName) ? defaultLabel : NewRoiName.Trim();
        var normalized = CloneRoi(roi);
        normalized.Name = name;
        if (string.IsNullOrWhiteSpace(normalized.Id))
            normalized.Id = defaultLabel;
        _drawnRois.Add(normalized);
        SelectedRoi = normalized;
        NewRoiName = BuildDefaultRoiLabel(_drawnRois.Count + 1, roi.Kind);
        UpdateSelectedRoiStats();
        CommandManager.InvalidateRequerySuggested();
        _log.AddLog($"ROI 추가: {normalized.Name} [{normalized.Id}] ({normalized.Kind})");
    }

    public void RemoveSelectedRoi()
    {
        if (SelectedRoi == null) return;
        int idx = _drawnRois.IndexOf(SelectedRoi);
        if (idx < 0) return;
        _drawnRois.RemoveAt(idx);
        SelectedRoi = idx < _drawnRois.Count ? _drawnRois[idx] : _drawnRois.LastOrDefault();
        UpdateSelectedRoiStats();
        CommandManager.InvalidateRequerySuggested();
    }

    public void ClearDrawnRois()
    {
        _drawnRois.Clear();
        SelectedRoi = null;
        SelectedRoiStatsText = "ROI 통계 없음";
        CommandManager.InvalidateRequerySuggested();
    }

    private bool[] BuildUnionMaskFromDrawnRois(int rows, int cols)
    {
        var union = new bool[rows * cols];
        foreach (var roi in _drawnRois)
        {
            var one = GridRoiMaskBuilder.BuildMask(rows, cols, roi);
            int n = Math.Min(union.Length, one.Length);
            for (int i = 0; i < n; i++)
                union[i] |= one[i];
        }
        return union;
    }

    private bool[]? BuildMaskFromRoiFile(FloatGrid grid)
    {
        var rois = LoadRoisFromFile(RoiJsonPath, grid);
        if (rois.Count == 0) return null;
        var union = new bool[grid.Length];
        foreach (var roi in rois)
        {
            var one = GridRoiMaskBuilder.BuildMask(grid.Rows, grid.Cols, roi);
            int n = Math.Min(union.Length, one.Length);
            for (int i = 0; i < n; i++)
                union[i] |= one[i];
        }
        return union;
    }

    private static List<GridRoiDefinition> LoadRoisFromFile(string path, FloatGrid grid)
    {
        try
        {
            var cat = GridRoiCatalog.Load(path);
            cat.ValidateAgainstGrid(grid);
            return cat.EnumerateDefinitions().ToList();
        }
        catch
        {
            var roi = GridRoiJsonIO.ReadSingleRoi(path, out var rr, out var cc);
            GridRoiJsonIO.ValidateReferenceShape(rr, cc, grid);
            return new List<GridRoiDefinition> { roi };
        }
    }

    private static List<GridRoiDefinition> LoadRoisFromFileWithoutGrid(string path)
    {
        try
        {
            var cat = GridRoiCatalog.Load(path);
            return cat.EnumerateDefinitions().ToList();
        }
        catch
        {
            var roi = GridRoiJsonIO.ReadSingleRoi(path, out _, out _);
            return new List<GridRoiDefinition> { roi };
        }
    }

    private static RoiCatalogEntryDto ToCatalogEntry(GridRoiDefinition roi) => new()
    {
        Id = roi.Id,
        Kind = roi.Kind.ToString(),
        Name = roi.Name,
        X = roi.X,
        Y = roi.Y,
        Width = roi.Width,
        Height = roi.Height,
        C0 = roi.C0,
        R0 = roi.R0,
        C1 = roi.C1,
        R1 = roi.R1,
        C2 = roi.C2,
        R2 = roi.R2,
        CenterCol = roi.CenterCol,
        CenterRow = roi.CenterRow,
        RadiusPixels = roi.RadiusPixels
    };

    private static GridRoiDefinition CloneRoi(GridRoiDefinition roi) => new()
    {
        Id = roi.Id,
        Name = roi.Name,
        Kind = roi.Kind,
        X = roi.X,
        Y = roi.Y,
        Width = roi.Width,
        Height = roi.Height,
        C0 = roi.C0,
        R0 = roi.R0,
        C1 = roi.C1,
        R1 = roi.R1,
        C2 = roi.C2,
        R2 = roi.R2,
        CenterCol = roi.CenterCol,
        CenterRow = roi.CenterRow,
        RadiusPixels = roi.RadiusPixels
    };

    private static string BuildDefaultRoiLabel(int roiNumber, RoiKind kind) =>
        $"roi{Math.Max(1, roiNumber):000}_{kind.ToString().ToLowerInvariant()}";

    private static void NormalizeRoiIdentity(GridRoiDefinition roi, int roiNumber)
    {
        string label = BuildDefaultRoiLabel(roiNumber, roi.Kind);
        if (string.IsNullOrWhiteSpace(roi.Id))
            roi.Id = label;
        if (string.IsNullOrWhiteSpace(roi.Name))
            roi.Name = roi.Id;
    }

    public void UpdateDiffCursorInfo(int col, int row)
    {
        if (_lastResult == null || col < 0 || row < 0 || col >= _lastResult.Diff.Cols || row >= _lastResult.Diff.Rows)
        {
            DiffCursorInfo = "x=-, y=-, value=-, norm=-";
            return;
        }

        double v = _lastResult.Diff[row, col];
        string vText = FloatGrid.IsFinite(v) ? v.ToString("F6", CultureInfo.InvariantCulture) : "NaN";
        string normText = "NaN";
        if (FloatGrid.IsFinite(v) && _diffDisplayMax > _diffDisplayMin)
        {
            double n = Math.Clamp((v - _diffDisplayMin) / (_diffDisplayMax - _diffDisplayMin), 0, 1);
            normText = n.ToString("F4", CultureInfo.InvariantCulture);
        }
        DiffCursorInfo = $"x={col}, y={row}, value={vText}, norm={normText}";
    }

    public void ResetDiffCursorInfo() => DiffCursorInfo = "x=-, y=-, value=-, norm=-";

    public void UpdateSelectedRoiStats()
    {
        if (_lastResult == null || SelectedRoi == null)
        {
            SelectedRoiStatsText = "ROI 통계 없음";
            return;
        }

        var mask = GridRoiMaskBuilder.BuildMask(_lastResult.Diff.Rows, _lastResult.Diff.Cols, SelectedRoi);
        int roiPixels = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i]) roiPixels++;
        }

        if (roiPixels <= 0)
        {
            SelectedRoiStatsText = $"선택 ROI '{SelectedRoi.Name}' | ROI 픽셀=0 (격자와 겹침 없음)";
            return;
        }

        var s = GridStatisticsService.Compute(_lastResult.Diff, mask, null, null);
        if (s.Count <= 0)
        {
            SelectedRoiStatsText =
                $"선택 ROI '{SelectedRoi.Name}' | ROI 픽셀={roiPixels}, 유효 픽셀=0 (NaN·Inf 등)";
            return;
        }

        int invalidInRoi = roiPixels - s.Count;
        string invalidHint = invalidInRoi > 0 ? $", 비유효(ROI 내)={invalidInRoi}" : "";
        SelectedRoiStatsText =
            $"선택 ROI '{SelectedRoi.Name}' \n ROI 픽셀={roiPixels}, 유효 픽셀={s.Count}{invalidHint}, Finite Ratio = {s.Count/roiPixels} \n Mean={s.Mean:F6}, Std={s.Std:F6}, Min={s.Min:F6}, Max={s.Max:F6}, Count: {s.Count}";
    }

    private async Task RunBatchFolderAsync()
    {
        try
        {
            if (!Directory.Exists(BatchFolder))
                throw new DirectoryNotFoundException($"배치 폴더를 찾을 수 없습니다: {BatchFolder}");

            PickOutputFolder();
            if (string.IsNullOrWhiteSpace(OutputFolder) || !Directory.Exists(OutputFolder))
                return;

            var opt = new GridBatchOptions
            {
                Alignment = BuildAlign(),
                PreprocessA = BuildPre(),
                PreprocessB = BuildPre(),
                Combine = BuildCombine(),
                Visualization = new GridVisualizationOptions(),
                InputColormapId = CmapInput,
                DiffColormapId = CmapDiff
            };
            if (File.Exists(CatalogPath))
                opt.Catalog = GridRoiCatalog.Load(CatalogPath);
            else if (_drawnRois.Count > 0)
                opt.Catalog = BuildCatalogFromCurrentRois();
            string folderName = new DirectoryInfo(BatchFolder).Name;
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string outDir = Path.Combine(OutputFolder, $"{folderName}_{stamp}");
            var batchFolder = BatchFolder;
            BeginGridBusy("배치 처리 중…");
            try
            {
                await Task.Run(() => GridBatchService.RunFolderParallel(batchFolder, outDir, opt)).ConfigureAwait(true);
                _log.AddLog($"Batch finished → {outDir}");
            }
            finally
            {
                EndGridBusy();
            }
        }
        catch (Exception ex)
        {
            _log.AddLog($"Batch error: {ex.Message}");
            System.Windows.MessageBox.Show(ex.Message);
        }
    }

    private async Task SummarizeDiffsAsync()
    {
        BeginGridBusy("Diff ROI 요약…");
        try
        {
            var cat = File.Exists(CatalogPath) ? GridRoiCatalog.Load(CatalogPath) : BuildCatalogFromCurrentRois();
            string outCsv = string.IsNullOrWhiteSpace(SummarizeOutCsv)
                ? Path.Combine(BatchFolder, "diff_roi_summary.csv")
                : SummarizeOutCsv;
            var folder = BatchFolder;
            await Task.Run(() => GridBatchService.SummarizeDiffCsvsWithCatalog(folder, cat, outCsv, PostLog)).ConfigureAwait(true);
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show(ex.Message);
        }
        finally
        {
            EndGridBusy();
        }
    }

    private GridRoiCatalog BuildCatalogFromCurrentRois()
    {
        if (_drawnRois.Count == 0)
            throw new InvalidOperationException("현재 워크벤치에 ROI가 없습니다. 먼저 ROI를 그리거나 ROI JSON을 불러오세요.");

        for (int i = 0; i < _drawnRois.Count; i++)
            NormalizeRoiIdentity(_drawnRois[i], i + 1);

        return new GridRoiCatalog
        {
            // 참조 크기를 0으로 두면 각 diff CSV의 크기 검사 없이 동일 ROI 좌표를 적용합니다.
            ReferenceRows = 0,
            ReferenceCols = 0,
            Rois = _drawnRois.Select(ToCatalogEntry).ToList()
        };
    }

    /// <summary>Diff 미리보기 갱신(비동기). 표시 min/max가 둘 다 유효하면 그 범위로, 아니면 결합 옵션 기반 자동 범위.</summary>
    public void RefreshDiffPreview() => _ = ScheduleRefreshDiffPreviewAsync();

    private async Task ScheduleRefreshDiffPreviewAsync()
    {
        var token = Interlocked.Increment(ref _diffPreviewToken);
        await RefreshDiffPreviewWorkerAsync(token, manageBusy: true).ConfigureAwait(true);
    }

    private async Task RefreshDiffPreviewWorkerAsync(int token, bool manageBusy = true)
    {
        if (_lastResult == null) return;
        if (manageBusy) BeginGridBusy("Diff 미리보기…");
        try
        {
            var diffGrid = _lastResult.Diff;
            var combine = BuildCombine();
            var vis = new GridVisualizationOptions();
            var minT = DiffDisplayMinText;
            var maxT = DiffDisplayMaxText;
            var mode = DiffPreviewMode;
            var cmap = CmapDiff;

            var (pngBytes, dMin, dMax) = await Task.Run(() =>
            {
                double umin = 0, umax = 0;
                var hasMin = double.TryParse(minT, NumberStyles.Any, CultureInfo.InvariantCulture, out umin);
                var hasMax = double.TryParse(maxT, NumberStyles.Any, CultureInfo.InvariantCulture, out umax);
                var manual = hasMin && hasMax && umax > umin;
                double dLo, dHi;
                if (manual)
                {
                    dLo = umin;
                    dHi = umax;
                }
                else
                    (dLo, dHi) = GridCombineService.GetDisplayRange(diffGrid, combine);

                if (mode == "그레이스케일")
                {
                    using var m = GridVisualizationService.ToGray8Preview(diffGrid, vis, dLo, dHi);
                    return (GridMatPng.EncodePng(m), dLo, dHi);
                }
                using var md = GridVisualizationService.RenderHeatmapBgra(diffGrid, dLo, dHi, cmap, vis);
                return (GridMatPng.EncodePng(md), dLo, dHi);
            }).ConfigureAwait(true);

            if (token != Volatile.Read(ref _diffPreviewToken))
                return;
            _diffDisplayMin = dMin;
            _diffDisplayMax = dMax;
            PreviewDiff = MatBitmapConverter.FromPngBytes(pngBytes);
        }
        finally
        {
            if (manageBusy) EndGridBusy();
        }
    }
}
