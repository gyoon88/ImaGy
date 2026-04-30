using MathNet.Numerics.Distributions;

namespace ImaGy.Grids;

public sealed record TwoSampleTestResult(
    string Name,
    double Statistic,
    double? DfBetween,
    double? DfWithin,
    double PValueTwoSided,
    string Note);

/// <summary>Diff 격자에서 두 ROI(또는 두 마스크)의 유한 값 표본에 대한 기초 검정.</summary>
public static class GridRoiHypothesisTests
{
    public static List<double> ExtractFiniteSample(FloatGrid grid, bool[]? mask)
    {
        var list = new List<double>();
        for (var i = 0; i < grid.Length; i++)
        {
            if (mask != null && (i >= mask.Length || !mask[i]))
                continue;
            var v = grid.Data[i];
            if (!FloatGrid.IsFinite(v))
                continue;
            list.Add(v);
        }
        return list;
    }

    public static IReadOnlyList<TwoSampleTestResult> RunAll(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        if (a.Count < 2 || b.Count < 2)
            return Array.Empty<TwoSampleTestResult>();

        var results = new List<TwoSampleTestResult>();
        if (TryPooledT(a, b, out var r1))
            results.Add(r1);
        if (TryWelchT(a, b, out var r2))
            results.Add(r2);
        if (TryOneWayAnovaTwoGroups(a, b, out var r3))
            results.Add(r3);
        if (TryMannWhitneyNormalApprox(a, b, out var r4))
            results.Add(r4);
        return results;
    }

    private static bool TryPooledT(IReadOnlyList<double> a, IReadOnlyList<double> b, out TwoSampleTestResult r)
    {
        r = default!;
        var n1 = a.Count;
        var n2 = b.Count;
        if (n1 < 2 || n2 < 2)
            return false;
        var m1 = a.Average();
        var m2 = b.Average();
        var v1 = SampleVariance(a, m1);
        var v2 = SampleVariance(b, m2);
        var df = n1 + n2 - 2;
        var sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df;
        if (sp2 <= 0 || double.IsNaN(sp2))
            return false;
        var se = Math.Sqrt(sp2 * (1.0 / n1 + 1.0 / n2));
        if (se < 1e-30)
            return false;
        var t = (m1 - m2) / se;
        var dist = new StudentT(0, 1, df);
        var p = 2 * (1 - dist.CumulativeDistribution(Math.Abs(t)));
        r = new TwoSampleTestResult("독립 표본 t (등분산 가정)", t, null, df, ClampP(p),
            "sp² 풀링, df=n₁+n₂−2. 등분산 가정이 맞지 않으면 Welch를 참고하세요.");
        return true;
    }

    private static bool TryWelchT(IReadOnlyList<double> a, IReadOnlyList<double> b, out TwoSampleTestResult r)
    {
        r = default!;
        var n1 = a.Count;
        var n2 = b.Count;
        if (n1 < 2 || n2 < 2)
            return false;
        var m1 = a.Average();
        var m2 = b.Average();
        var v1 = SampleVariance(a, m1);
        var v2 = SampleVariance(b, m2);
        var s1n = v1 / n1;
        var s2n = v2 / n2;
        var se = Math.Sqrt(s1n + s2n);
        if (se < 1e-30 || double.IsNaN(se))
            return false;
        var t = (m1 - m2) / se;
        var num = s1n + s2n;
        var den = s1n * s1n / Math.Max(1, n1 - 1) + s2n * s2n / Math.Max(1, n2 - 1);
        var df = num * num / den;
        if (df < 1e-6 || double.IsNaN(df))
            return false;
        var dist = new StudentT(0, 1, df);
        var p = 2 * (1 - dist.CumulativeDistribution(Math.Abs(t)));
        r = new TwoSampleTestResult("Welch t (이분산)", t, null, df, ClampP(p),
            "Satterthwaite 근사 자유도. 등분산을 가정하지 않습니다.");
        return true;
    }

    private static bool TryOneWayAnovaTwoGroups(IReadOnlyList<double> a, IReadOnlyList<double> b, out TwoSampleTestResult r)
    {
        r = default!;
        var n1 = a.Count;
        var n2 = b.Count;
        var n = n1 + n2;
        if (n1 < 2 || n2 < 2)
            return false;
        var all = a.Concat(b).ToArray();
        var mg = all.Average();
        var m1 = a.Average();
        var m2 = b.Average();
        var ssBetween = n1 * (m1 - mg) * (m1 - mg) + n2 * (m2 - mg) * (m2 - mg);
        var ssWithin = a.Sum(x => (x - m1) * (x - m1)) + b.Sum(x => (x - m2) * (x - m2));
        var dfB = 1;
        var dfW = n - 2;
        if (dfW < 1 || ssWithin <= 0)
            return false;
        var msB = ssBetween / dfB;
        var msW = ssWithin / dfW;
        if (msW < 1e-30)
            return false;
        var f = msB / msW;
        var fd = new FisherSnedecor(dfB, dfW);
        var p = 1 - fd.CumulativeDistribution(f);
        r = new TwoSampleTestResult("일원 배치 ANOVA (그룹 2, F)", f, dfB, dfW, ClampP(p),
            "F=MS_between/MS_within. p는 F 분포 상위 꼬리(일반적인 ANOVA 보고). 그룹 2개·등분산일 때 F=t²입니다.");
        return true;
    }

    /// <summary>Mann–Whitney U, 표준 정규 근사(동순위는 평균 순위).</summary>
    private static bool TryMannWhitneyNormalApprox(IReadOnlyList<double> a, IReadOnlyList<double> b, out TwoSampleTestResult r)
    {
        r = default!;
        var n1 = a.Count;
        var n2 = b.Count;
        if (n1 < 4 || n2 < 4)
            return false;

        var tagged = new List<(double v, int g)>(n1 + n2);
        foreach (var x in a) tagged.Add((x, 1));
        foreach (var x in b) tagged.Add((x, 2));
        tagged.Sort((x, y) => x.v.CompareTo(y.v));

        var ranks = new double[tagged.Count];
        for (var i = 0; i < tagged.Count;)
        {
            var j = i;
            while (j < tagged.Count && Math.Abs(tagged[j].v - tagged[i].v) < 1e-15 * (1 + Math.Abs(tagged[i].v)))
                j++;
            var avgRank = (i + j + 1) / 2.0;
            for (var k = i; k < j; k++)
                ranks[k] = avgRank;
            i = j;
        }

        double r1 = 0;
        for (var i = 0; i < tagged.Count; i++)
        {
            if (tagged[i].g == 1)
                r1 += ranks[i];
        }

        var u1 = r1 - n1 * (n1 + 1) / 2.0;
        var mu = n1 * n2 / 2.0;
        var sigma = Math.Sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0);
        if (sigma < 1e-15)
            return false;
        var z = (u1 - mu) / sigma;
        var nd = new Normal(0, 1);
        var p = 2 * (1 - nd.CumulativeDistribution(Math.Abs(z)));
        r = new TwoSampleTestResult("Mann–Whitney U (정규 근사)", u1, null, null, ClampP(p),
            "U₁=ΣR₁−n₁(n₁+1)/2. 동순위 많으면 정확 검정과 차이날 수 있습니다.");
        return true;
    }

    private static double SampleVariance(IReadOnlyList<double> x, double mean)
    {
        if (x.Count < 2)
            return double.NaN;
        var s = 0.0;
        foreach (var v in x)
            s += (v - mean) * (v - mean);
        return s / (x.Count - 1);
    }

    private static double ClampP(double p)
    {
        if (double.IsNaN(p) || double.IsInfinity(p))
            return double.NaN;
        return Math.Clamp(p, 0, 1);
    }
}
