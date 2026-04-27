# ImaGy — 프로젝트 개요

Windows용 **이미지 처리 데스크톱 앱**과, CSV 기반 **2D 격자(그리드) 파이프라인** 도구를 한 솔루션에서 빌드합니다. UI는 **WPF**, 고성능 필터·FFT·템플릿 매칭 등은 **C++ DLL**에서 처리하고, C#은 **C++/CLI 래퍼**를 통해 네이티브 API를 호출합니다.

---

## 1. 솔루션 구성

| 프로젝트 | 종류 | 역할 |
|-----------|------|------|
| **ImaGy** | C# WPF (`WinExe`) | 메인 앱, MVVM, 이미지 뷰·처리 UI |
| **ImaGyNative** | C++ DLL | `NativeCore` — CPU(SSE)·**CUDA** 커널, FFT(cuFFT) 등 |
| **ImaGyWrapper** | C++/CLI DLL (`/clr:netcore`) | `ImaGy::Wrapper::NativeProcessor` — 관리 코드 ↔ 네이티브 포인터 브리지 |
| **ImaGy.Grids** | C# 라이브러리 | CSV 격자 읽기/정렬/전처리/결합/시각화, **OpenCvSharp**, **ScottPlot** |
| **ImaGy.GridCli** | C# 콘솔 (`Exe`) | `ImaGy.Grids` 참조 CLI(배치·실험용) |

의존 방향 요약:

`ImaGy` → `ImaGy.Grids`, `ImaGyWrapper` + 출력 시 `ImaGyNative.dll` 복사  
`ImaGyWrapper` → `ImaGyNative`  
`ImaGy.GridCli` → `ImaGy.Grids`

---

## 2. 런타임·플랫폼

| 항목 | 값 |
|------|-----|
| **메인 / Grids / CLI 타깃** | `net8.0-windows10.0.26100.0` |
| **최소 OS(메인 프로젝트)** | `SupportedOSPlatformVersion` 10.0.26100.0 |
| **메인 앱 플랫폼** | `x64` (`ImaGy.csproj`의 `<Platforms>x64</Platforms>`) |
| **UI** | WPF (`UseWPF`), WinForms 상호 운용 (`UseWindowsForms`) — 파일 대화상자·클립보드 등 |
| **래퍼 타깃** | `net8.0-windows10.0.26100.0`, `CLRSupport=NetCore` |

---

## 3. 네이티브 (ImaGyNative)

| 항목 | 내용 |
|------|------|
| **산출물** | `ImaGyNative.dll` (동적 라이브러리) |
| **Windows SDK** | `10.0.26100.0` (`WindowsTargetPlatformVersion`) |
| **플랫폼 툴셋** | **v142** (Visual Studio 2019 C++ 빌드 도구) — `ImaGyNative.vcxproj` 기준 |
| **CUDA** | **12.6** — `CUDA 12.6.props` / `CUDA 12.6.targets`, 기본 경로 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` |
| **링크 라이브러리** | `cufft.lib`, `cudart_static.lib` |
| **병렬** | OpenMP (`<OpenMPSupport>true</OpenMPSupport>`) |
| **GPU 소스** | `CudaKernel.cu`, `CudaColorKernel.cu` |
| **CPU** | `NativeCoreSse.cpp` 등 SSE 경로, `CPUImageProcessor.cpp` |

다른 PC에서 빌드할 때는 CUDA 설치 경로가 다르면 `ImaGyNative.vcxproj`의 `IncludePath` / `LibraryPath`를 환경에 맞게 수정해야 합니다.

---

## 4. C++/CLI (ImaGyWrapper)

| 항목 | 내용 |
|------|------|
| **역할** | `System::IntPtr` 등으로 픽셀 버퍼 포인터를 받아 `ImaGyNative::NativeCore::*` 호출 |
| **타깃 프레임워크** | `net8.0-windows10.0.26100.0` |
| **툴셋** | **v143** (VS 2022) — .NET 8의 `IntPtr`/generic math 메타데이터와 호환되도록 권장 |
| **프레임워크 참조** | `Microsoft.WindowsDesktop.App.WPF` (`FrameworkReference`) |
| **관리 참조** | `#using <System.dll>` 등 (`ImaGyWrapper.h`) |

빌드 시 NuGet 복원으로 `obj\project.assets.json`이 필요할 수 있습니다 (`msbuild /t:Restore`).

---

## 5. C# 패키지·라이브러리

### ImaGy (메인)

| 패키지 | 버전(프로젝트 기준) |
|--------|---------------------|
| **Microsoft.Xaml.Behaviors.Wpf** | 1.1.135 |

### ImaGy.Grids

| 패키지 | 버전(프로젝트 기준) |
|--------|---------------------|
| **OpenCvSharp4.Windows** | 4.11.0.20250507 |
| **ScottPlot** | 5.0.55 |
| **System.Text.Json** | 9.0.0 |

격자 시각화·PNG 인코딩·CLI는 이 프로젝트를 통해 **OpenCV C# 바인딩**과 **ScottPlot**에 의존합니다.

### ImaGy.GridCli

별도 NuGet 없음 — `ImaGy.Grids`만 참조.

---

## 6. 빌드·실행 시 주의

1. **전체 솔루션**은 Visual Studio에서 **x64**, **ImaGyNative**·**ImaGyWrapper**·**ImaGy** 순으로 의존성이 맞게 빌드하는 것이 안전합니다. `dotnet build`만으로는 C++/CLI·CUDA 프로젝트가 생략되거나 실패할 수 있습니다.
2. 실행 폴더에 **`ImaGyNative.dll`**(및 CUDA 런타임이 필요한 경우 NVIDIA 드라이버)이 있어야 합니다. 메인 프로젝트는 `x64\$(Configuration)\ImaGyNative.dll`을 출력으로 복사하도록 구성되어 있습니다.
3. **README**에는 영상 처리 알고리즘에 OpenCV 사용 금지라는 제약이 적혀 있으나, **격자 모듈(`ImaGy.Grids`)은 OpenCvSharp 기반**입니다. 문서와 실제 정책이 다르면 README를 업데이트하는 것이 좋습니다.

---

## 7. 주요 기능 영역 (코드 기준)

- **이미지**: 대비·이진화·K-means·평활화·히스토그램, 소벨·라플라시안·FFT·주파수 필터, 블러, 형태학, NCC/SAD/SSD 등 — `ImaGyWrapper` / `ImaGyNative`.
- **격자 워크벤치**: CSV A/B 로드, 정렬·전처리·결합 파이프라인, 히트맵·ScottPlot PNG, 히스토그램 창, 배치 — `ImaGy.Grids` + `GridWorkbenchWindow`.
- **UI**: 다크 테마 리소스는 `App.xaml`의 브러시·`Window`/`GroupBox`/`Button` 등 스타일에 정의되어 있습니다.

---

## 8. 관련 문서

- 저장소 루트 **`README.md`**: 초기 기능·UI 명세(스펙 문서 성격).
- 본 파일: **현재 레포 구성·버전** 위주의 기술 요약(빌드/의존성 점검용).

버전이 바뀌면 각 `.csproj` / `.vcxproj`를 기준으로 이 문서를 갱신하면 됩니다.
