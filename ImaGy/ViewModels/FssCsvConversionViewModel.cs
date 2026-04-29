using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Input;
using ImaGy.Grids;
using ImaGy.Services;
using ImaGy.ViewModels.Commands;

namespace ImaGy.ViewModels;

public sealed class FssCsvConversionViewModel : BaseViewModel
{
    private readonly LoggingService _log;
    private bool _isRawMode = true;
    private string _binXText = "";
    private string _binYText = "";
    private string _searchXText = "1";
    private string _searchYText = "1";
    private string _coresText = "5";
    private string _pairCountText = "4";
    private bool _saveIntensity = true;
    private bool _saveDistance = true;
    private bool _saveQuality = true;
    private string _statusText = "CSV 파일을 추가한 뒤 변환을 실행하세요.";
    private double _progressPercent;
    private bool _isConverting;

    public ObservableCollection<string> FilePaths { get; } = new();

    public double ProgressPercent
    {
        get => _progressPercent;
        set => SetProperty(ref _progressPercent, value);
    }

    public bool IsConverting
    {
        get => _isConverting;
        private set
        {
            if (!SetProperty(ref _isConverting, value))
                return;
            CommandManager.InvalidateRequerySuggested();
        }
    }

    public bool IsRawMode
    {
        get => _isRawMode;
        set
        {
            if (!SetProperty(ref _isRawMode, value))
                return;
            OnPropertyChanged(nameof(IntensityEnabled));
        }
    }

    public bool IsInterpMode
    {
        get => !_isRawMode;
        set => IsRawMode = !value;
    }

    public bool IntensityEnabled => IsRawMode;

    public string BinXText
    {
        get => _binXText;
        set => SetProperty(ref _binXText, value);
    }

    public string BinYText
    {
        get => _binYText;
        set => SetProperty(ref _binYText, value);
    }

    public string SearchXText
    {
        get => _searchXText;
        set => SetProperty(ref _searchXText, value);
    }

    public string SearchYText
    {
        get => _searchYText;
        set => SetProperty(ref _searchYText, value);
    }

    public string CoresText
    {
        get => _coresText;
        set => SetProperty(ref _coresText, value);
    }

    public string PairCountText
    {
        get => _pairCountText;
        set => SetProperty(ref _pairCountText, value);
    }

    public bool SaveIntensity
    {
        get => _saveIntensity;
        set => SetProperty(ref _saveIntensity, value);
    }

    public bool SaveDistance
    {
        get => _saveDistance;
        set => SetProperty(ref _saveDistance, value);
    }

    public bool SaveQuality
    {
        get => _saveQuality;
        set => SetProperty(ref _saveQuality, value);
    }

    public string StatusText
    {
        get => _statusText;
        set => SetProperty(ref _statusText, value);
    }

    public ICommand AddFilesCommand { get; }
    public ICommand ClearFilesCommand { get; }
    public ICommand RunConversionCommand { get; }

    public FssCsvConversionViewModel(LoggingService log)
    {
        _log = log;
        AddFilesCommand = new RelayCommand(AddFiles);
        ClearFilesCommand = new RelayCommand(() =>
        {
            FilePaths.Clear();
            CommandManager.InvalidateRequerySuggested();
        });
        RunConversionCommand = new RelayCommand(() => _ = RunConversionAsync(), () => FilePaths.Count > 0 && !IsConverting);
    }

    private void AddFiles()
    {
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "CSV (*.csv)|*.csv|모든 파일 (*.*)|*.*",
            Multiselect = true
        };
        if (dlg.ShowDialog() != true)
            return;
        foreach (var p in dlg.FileNames)
        {
            if (!FilePaths.Contains(p))
                FilePaths.Add(p);
        }
        StatusText = $"{FilePaths.Count}개 파일";
        CommandManager.InvalidateRequerySuggested();
    }

    private async Task RunConversionAsync()
    {
        if (IsConverting)
            return;
        if (FilePaths.Count == 0)
        {
            System.Windows.MessageBox.Show("먼저 CSV 파일을 추가하세요.", "FSS CSV 변환", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        if (IsRawMode)
        {
            if (!SaveIntensity && !SaveDistance && !SaveQuality)
            {
                System.Windows.MessageBox.Show("저장할 항목(Intensity / Distance / Quality)을 하나 이상 선택하세요.", "FSS CSV 변환",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
        }
        else if (!SaveDistance && !SaveQuality)
        {
            System.Windows.MessageBox.Show("Interpolation 모드에서는 Distance 또는 Quality를 하나 이상 선택하세요.", "FSS CSV 변환",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!int.TryParse(CoresText.Trim(), out var cores) || cores < 1)
            cores = 5;
        if (!int.TryParse(PairCountText.Trim(), out var pairCount) || pairCount < 1)
            pairCount = 4;
        pairCount = Math.Clamp(pairCount, 1, 4);

        if (!int.TryParse(SearchXText.Trim(), out var sx) || sx < 0)
            sx = 1;
        if (!int.TryParse(SearchYText.Trim(), out var sy) || sy < 0)
            sy = 1;

        IsConverting = true;
        ProgressPercent = 0;
        try
        {
            foreach (var path in FilePaths.ToList())
            {
                var fileLabel = Path.GetFileName(path);
                var progress = CreateFssProgressReporter(fileLabel);
                try
                {
                    if (IsRawMode)
                        await Task.Run(() => RunOneRaw(path, sx, sy, pairCount, cores, progress)).ConfigureAwait(true);
                    else
                        await Task.Run(() => RunOneInterp(path, pairCount, progress)).ConfigureAwait(true);
                    _log.AddLog($"[FSS] 완료: {fileLabel}");
                }
                catch (Exception ex)
                {
                    _log.AddLog($"[FSS] 오류 {fileLabel}: {ex.Message}");
                    System.Windows.MessageBox.Show($"{fileLabel}\n{ex.Message}", "FSS CSV 변환 오류",
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }

            StatusText = "변환 작업이 끝났습니다.";
            System.Windows.MessageBox.Show("선택한 파일 처리가 완료되었습니다.", "FSS CSV 변환", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        finally
        {
            IsConverting = false;
            ProgressPercent = 0;
        }
    }

    /// <summary>UI 스레드에서 생성해야 Progress 콜백이 UI로 마샬됩니다.</summary>
    private IProgress<string> CreateFssProgressReporter(string fileLabel) =>
        new Progress<string>(msg =>
        {
            if (msg.StartsWith("P:", StringComparison.Ordinal)
                && int.TryParse(msg.AsSpan(2), NumberStyles.Integer, CultureInfo.InvariantCulture, out var p))
                ProgressPercent = Math.Clamp(p, 0, 100);
            else if (msg.StartsWith("S:", StringComparison.Ordinal))
            {
                ProgressPercent = Math.Max(ProgressPercent, 99);
                StatusText = $"{fileLabel}: {msg[2..]}";
            }
            else
                StatusText = $"{fileLabel}: {msg}";
            _log.AddLog($"[FSS] {fileLabel} {msg}");
        });

    private void RunOneRaw(string filePath, int searchX, int searchY, int pairCount, int cores, IProgress<string> progress)
    {
        var layout = FssAreaScanConverter.DetectLayout(filePath);
        if (layout.Shape == FssAreaScanConverter.CsvShapeKind.Wide)
        {
            var flags = new FssAreaScanConverter.RawSaveFlags
            {
                IntensityCsv = SaveIntensity,
                DistanceCsv = SaveDistance,
                QualityCsv = SaveQuality,
                IntensityPng = SaveIntensity,
                DistancePng = SaveDistance,
                QualityPng = SaveQuality
            };
            FssAreaScanConverter.ConvertRaw(filePath, searchX, searchY, pairCount, 0, 0, cores, flags, progress);
            return;
        }

        var peek = FssAreaScanConverter.PeekLongGrid(filePath);
        if (peek.ErrorMessage != null)
            throw new InvalidOperationException(peek.ErrorMessage);

        var infW = peek.InferredWidth;
        var infH = peek.InferredHeight;
        var rows = peek.RowCount;

        var binXEmpty = string.IsNullOrWhiteSpace(BinXText);
        var binYEmpty = string.IsNullOrWhiteSpace(BinYText);
        int useW, useH;
        if (binXEmpty && binYEmpty)
        {
            useW = infW;
            useH = infH;
        }
        else if (!TryParsePositiveBin(BinXText, out var userW) | !TryParsePositiveBin(BinYText, out var userH))
        {
            if (binXEmpty || binYEmpty)
            {
                useW = infW;
                useH = infH;
            }
            else
                throw new InvalidOperationException("X/Y 빈 개수는 비우면 자동, 또는 둘 다 양의 정수로 입력하세요.");
        }
        else
        {
            if (userW != infW || userH != infH)
            {
                var r = System.Windows.Application.Current.Dispatcher.Invoke(() =>
                    System.Windows.MessageBox.Show(
                        $"좌표 패턴으로 추정된 그리드: 가로 {infW} × 세로 {infH} (데이터 행 수 {rows})\n" +
                        $"입력한 빈 수: 가로 {userW} × 세로 {userH}\n\n" +
                        "어느 크기로 변환할까요?\n\n" +
                        "예 = 추정값 사용\n" +
                        "아니오 = 입력값 사용\n" +
                        "취소 = 이 파일 건너뛰기",
                        "그리드 크기 불일치",
                        MessageBoxButton.YesNoCancel,
                        MessageBoxImage.Question,
                        MessageBoxResult.Yes));
                if (r == MessageBoxResult.Cancel)
                    return;
                if (r == MessageBoxResult.Yes)
                {
                    useW = infW;
                    useH = infH;
                }
                else
                {
                    useW = userW;
                    useH = userH;
                }
            }
            else
            {
                useW = userW;
                useH = userH;
            }
        }

        if (useW * useH != rows)
        {
            var r2 = System.Windows.Application.Current.Dispatcher.Invoke(() =>
                System.Windows.MessageBox.Show(
                    $"선택한 그리드 총 점 수는 {useW * useH}이고, 데이터 행 수는 {rows}입니다.\n" +
                    "Python 스크립트와 같이 부족하면 마지막 행을 복제해 패딩하고, 초과 행은 잘라 처리합니다.\n\n계속할까요?",
                    "행 수와 그리드 크기",
                    MessageBoxButton.OKCancel,
                    MessageBoxImage.Warning));
            if (r2 != MessageBoxResult.OK)
                return;
        }

        var flags2 = new FssAreaScanConverter.RawSaveFlags
        {
            IntensityCsv = SaveIntensity,
            DistanceCsv = SaveDistance,
            QualityCsv = SaveQuality,
            IntensityPng = SaveIntensity,
            DistancePng = SaveDistance,
            QualityPng = SaveQuality
        };
        FssAreaScanConverter.ConvertRaw(filePath, searchX, searchY, pairCount, useW, useH, cores, flags2, progress);
    }

    private static bool TryParsePositiveBin(string text, out int v)
    {
        v = 0;
        if (string.IsNullOrWhiteSpace(text))
            return false;
        return int.TryParse(text.Trim(), out v) && v > 0;
    }

    private void RunOneInterp(string filePath, int pairCount, IProgress<string> progress)
    {
        var flags = new FssAreaScanConverter.InterpSaveFlags
        {
            DistanceCsv = SaveDistance,
            QualityCsv = SaveQuality,
            DistancePng = SaveDistance,
            QualityPng = SaveQuality
        };
        FssAreaScanConverter.ConvertInterp(filePath, pairCount, flags, progress);
    }
}
