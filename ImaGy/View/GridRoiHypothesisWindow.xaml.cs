using System.Windows;
using ImaGy.ViewModels;

namespace ImaGy.View;

public partial class GridRoiHypothesisWindow
{
    public GridRoiHypothesisWindow()
    {
        InitializeComponent();
    }

    /// <summary>Owner 위에 검정 창을 띄우고 포그라운드로 가져옵니다.</summary>
    public static void ShowForWorkbench(Window owner, GridWorkbenchViewModel workbench)
    {
        var w = new GridRoiHypothesisWindow
        {
            Owner = owner,
            DataContext = new GridRoiHypothesisViewModel(workbench),
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        w.Show();
        w.Activate();
    }

    private void Close_Click(object sender, RoutedEventArgs e) => Close();
}
