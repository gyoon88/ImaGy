# ImaGy C++ Integration Plan

This document outlines the plan to integrate high-performance C++ code into the C# WPF application.

**Goal:**
Implement image processing functions in C++ and call them from C#. We are using Binarization as the first test case, hooking it up to the `Equalization` button for side-by-side comparison with the existing C# implementation.

**Strategy:**
1.  **`ImaGyNative`**: Native C++ DLL for core algorithms.
2.  **`ImaGyWrapper`**: A C++/CLI library to bridge C# and C++.
3.  **`ImaGy`**: The main C# application.

**Progress So Far:**
- User created the `ImaGyNative` and `ImaGyWrapper` projects.
- **`ImaGyNative`**: Created `NativeCore.h` and `NativeCore.cpp` with a basic C++ binarization function that operates on grayscale data.
- **`ImaGyWrapper`**: Updated `ImaGyWrapper.h` and `ImaGyWrapper.cpp` to handle the data marshalling (BitmapSource -> byte[] -> native pointer -> BitmapSource).
- **`ImaGy`**: Updated `ImageProcessor.cs` so that the `ApplyEqualization` method calls the C++ wrapper's binarization function.

---

### **Recent Progress & Resolutions**

**1. C++/CLI Wrapper (ImaGyWrapper) Setup:**
- The `ImaGyWrapper` project was confirmed to be a .NET 8 project.
- The compile error related to missing WPF types (`BitmapSource`) was resolved by adding `<FrameworkReference Include="Microsoft.WindowsDesktop.App.WPF" />` to `ImaGyWrapper.vcxproj`.

**2. C# to C++ Data Marshalling Refinement:**
- The `ImaGyNative` project's `NativeCore.h` and `NativeCore.cpp` were updated to accept `void* pixels, int width, int height, int stride, unsigned char threshold` for image data, allowing direct memory access.
- The `ImaGyWrapper` project's `NativeProcessor::ApplyBinarization` was simplified to directly pass `IntPtr` and image dimensions to the native C++ function, removing `BitmapSource` handling from the wrapper.
- A helper method `ProcessBitmapSourcePixels` was introduced in `ImageProcessor.cs` (C# Model) to centralize `BitmapSource` to `byte[]` conversion, pinning (`GCHandle`), and unpinning, making the `ApplyEqualization` method cleaner.

**3. Histogram Feature Implementation:**
- A `ServeHistogram.cs` class was implemented to calculate grayscale image histograms.
- `HistogramViewModel.cs` and `HistogramWindow.xaml`/`.xaml.cs` were created to display the histogram.
- The `MainViewModel.cs` was updated with a `ViewHistogramCommand` to open the `HistogramWindow`.
- Initial drawing issues in `HistogramWindow` (due to `ActualWidth`/`ActualHeight` being zero during early rendering) were resolved by redrawing the histogram on `Loaded` and `SizeChanged` events of the Canvas.
- **Further Histogram Improvements:** Implemented grid lines and axis labels in `HistogramWindow.xaml` and `HistogramWindow.xaml.cs` for better visualization. Refactored `HistogramViewModel` to observe `MainViewModel`'s image properties for self-contained data handling.

**4. C++ Native DLL Loading Issue Resolution:**
- The primary issue of `ImaGyWrapper.dll` failing to find `ImaGyNative.dll` at runtime was diagnosed.
- It was determined that `ImaGyNative.dll` was being built successfully in its own output directory (`$(SolutionDir)x64ackslash$(Configuration)\`) but was not being copied to the main `ImaGy` application's output directory (`ImaGy\bin\x64\Debug\net8.0-windows10.0.26100.0\`).
- This was resolved by explicitly adding `ImaGyNative.dll` as a content item to `ImaGy.csproj` with `CopyToOutputDirectory` set to `PreserveNewest`, ensuring it's copied to the final application output.

**5. Image Processing Functionalities (C++ & C# Interop):**
- All core image processing functions (Color/Contrast, Edge Detect, Blurring, Morphology), except FFT, have been implemented in `ImaGyNative` (C++) and integrated via `ImaGyWrapper` (C++/CLI).
- `ImageProcessor.cs` and `ImageProcessorSSE.cs` are correctly calling these native functions.

**6. Template Matching Implementation:**
- Added functionality to load a template image via a new menu option ("Open Template Image").
- Implemented a `TemplateImageViewer` and `TemplateImageViewerViewModel` to display the loaded template image in a separate window.
- The `NativeCore.h` and `NativeCore.cpp` signatures for `ApplyNCC`, `ApplySAD`, and `ApplySSD` have been updated to accept template image parameters.
- `BitmapProcessorHelper.ProcessTwoBitmapSourcePixels` was added to handle marshalling of two `BitmapSource` inputs (source and template) for native calls.
- `ImageProcessor.cs` and `MainViewModel.cs` have been updated to utilize the template image for `ApplyNCC`, `ApplySAD`, and `ApplySSD` operations, including asynchronous execution in `MainViewModel` to prevent UI freezing.

---

### **NEXT STEPS**
- 1) check the follow specification
- 2) refactoring the class structure 

## 1 Checkthe follow (Functional Specification)

### **2.1 사용자 인터페이스 (UI Layout)**

- **메인 창 레이아웃**
    - 상단: **메뉴바**
        - [파일] → 열기 **`(Ctrl+O)`**, 저장 **`(Ctrl+S)`**, 다른 이름으로 저장
        - [편집] → 실행 취소 **`(Ctrl+Z)`**, 다시 실행
        - [보기] → 히스토그램 보기
        - [도구]
            - 필터에 대한 파라미터 제공(가우시안 sigma & 필터 커널)
            - 이미지 자르기, ROI 선택
            - 이미지 복사, 붙여넣기
    - 좌측: **이미지 표시 영역 (Canvas 2개)**
        - 원본 이미지
        - 처리된 이미지
        - 스플리터(splitter)로 크기 조절 가능
    - 우측: **도킹 패널 (기능 영역)**
        - 버튼 그룹 (기본 조작 / 이미지 처리 / 고급 기능)
        - 리스트박스 (작업기록 표시)
        - 텍스트박스 (로그 출력 영역)
    - 하단: **상태바(Status Bar)**
        - 현재 파일명
        - 이미지 해상도
        - 처리 시간(ms)
        - 확대/축소 비율
- **히스토그램 창**
    - 계산된 히스토그램(분포) 출력
- **미니맵 창**
    - 확대 축소 된 영역에 대한 맵을 제공
    - ROI 선택 영역 제공

---

### **2.2 파일 처리 기능**

- **`PNG, JPEG, BMP`** 등 다양한 포맷 지원
- 저장 및 다른 이름으로 저장
- 단축키:
    - **`Ctrl+S`** → 저장
    - **`Ctrl+O`** → 열기
    - **`Ctrl+Z`** → 실행 취소
    - **`Ctrl+C`** → 이미지 복사
    - **`Ctrl+V`** → 이미지 붙여넣기
    - 

---

### **2.3 기본 이미지 조작**

- **확대/축소** (마우스 휠 + 드래그로 이동 지원)

---

### **2.4 영상 처리 기능**

- **형태학**
    - 팽창 **`(Morphological Dilation)`**
    - 수축 **`(Morphological Erosion)`**
- **색조 및 대비**
    - 평활화 **`(Equalization)`**
    - 이진화 **`(Binarization)`**
- **필터링**
    - 라플라스**`(Laplacian)`** 필터
    - 소벨**`(Sobel)`** 필터
    - 미분`(**Differential**)` 필터
    - 가우시안**`(Gaussian)`** 블러
    - 평균**`(Average)`** 블러
- **템플릿 매칭**
    - NCC (Normalized Cross Correlation)
    - SAD (Sum of Absolute Differences)
    - SSD (Sum of Squared Differences)
        - 이미지 매칭 시 다운사이징해서 진행
- **히스토그램**
    - RGB 분포 시각화
    - 누적 히스토그램

---

### **2.5 기타 기능**

- **히스토리 패널**
    - 실행한 연산 기록을 표시
    - 연산 실시 시간과 실행 시간 표시
- **로깅**
    - 처리 시각, 처리 시간(ms), 성공 여부 기록

---

## 3. 비기능 명세 (Non-Functional Specification)

- **성능**
    - SSE, CUDA 가속 지원
- **신뢰성**
    - 잘못된 파일 시 예외 처리
    - Undo/Redo 안정적 작동
- **사용성**
    - 직관적 버튼 배치
- **개발 제약**
    - Git Repository에 매일 커밋
    - 영상 처리 알고리즘은 직접 구현 (OpenCV 금지)

---

## 4. UI 명세 (WPF 기준)

### **메인 윈도우 구조**

```
+-------------------------------------------------------+
| [메뉴바: 파일 | 편집 | 보기 | 도움말 ]                 |
+-------------------------------------------------------+
| [이미지 뷰어 1]   | [이미지 뷰어 2]   | [도킹 패널]    |
| (원본 이미지)     | (처리된 이미지)   |                |
|                   |                  | [버튼 그룹]    |
|                   |                  | [히스토리]     |
|                   |                  | [로그창]       |
+-------------------------------------------------------+
| [상태바: 파일명 | 해상도 | 확대비율 | 처리시간]         |
+-------------------------------------------------------+

```

---

### **도킹 패널 상세**

- **버튼 그룹**
    - 기본 조작: 확대/축소, 자르기
        - 마우스 동작 제공
    - 영상 처리: 팽창, 수축, 평활화, 이진화
        - 이진화 임계값과 팽창 수축 커널 모양 정하기 제공
    - 필터링: 가우시안, 라플라스, 소벨
        - 필터 사이즈 고르는 파라미터(슬라이더 바 제공)
    - 히스토그램 보기
    - 템플릿 매칭 실행
- **히스토리 (리스트박스)**
    - `[10:22:15] 가우시안 필터 적용`
    - `[10:22:18] 이진화 적용`
- **로그 (텍스트박스, 멀티라인, ReadOnly)**
    - `[INFO] 파일 로드 완료: test.png`
    - `[INFO] 처리 시간: 25ms`


## 2. refactoring the class structure

### **/Models**

- ImageChannelUtils → 전처리 기능 (e.g. 채널 분리, 변환 등)
- ImageProcessor.cs → 처리 엔트리, 전략에 따라 위임
- FilteringProcessor.cs → Gaussian, Sobel, Laplacian
- MorphologyProcessor.cs → Erosion, Dilation 등
- MatchingProcessor.cs  → SAD, SSD, NCC 등
- ServeHistogram.cs   →  Histogram 계산
- ProcessTime.cs  →  처리 시간 측정용 모델
- MinimapModel.cs →  전체 이미지 + 현재 뷰 정보
- ParameterSet.cs →  필터 관련 파라미터 (e.g. Sigma, Kernel)
- ✅ ImageLayer.cs → 레이어 객체 -선택사항
- ✅ ImageLayerStack.cs → 레이어 스택 및 버퍼 - 선택사항
- ✅ RoiModel.cs → 관심영역(ROI)
- ✅ CropRegion.cs → 자르기 영역

---

### **/Services**

- ImageProcessingService.cs
- HistogramService.cs
- ImageChannelUtils.cs
- UndoRedoService.cs
- LoggingService.cs
- HistoryService.cs
- CsvExporter.cs
- ParameterSetService.cs
- MinimapService.cs
- ✅ ImageLayerManager.cs → 레이어 관리 기능
- ✅ RoiService.cs → ROI 생성 및 수정
- ✅ CropService.cs → 이미지 자르기 기능
- ✅ OverlayRenderer.cs → 이미지 위 표시 렌더링
- ✅ PixelInfoService.cs → 마우스 위치 픽셀 정보
- ✅ MacroExecutorService.cs → 매크로 실행 관리
- ✅ HistoryTreeService.cs → 분기형 히스토리 트리
- ✅ ClipboardImageService.cs → 이미지 복사/붙여넣기

---


### **/ViewModels**

- MainViewModel.cs  → 이미지 상태 총괄
- HistogramViewModel.cs -> histogram 관리
- ToolbarViewModel.cs
- MinimapViewModel.cs → 미니맵 뷰 관리
- ParameterViewModel.cs → 커널 크기, 시그마 등 슬라이더/입력창 바인딩
- ✅ LayerViewModel.cs → 레이어 추가/숨김/정렬 UI
- ✅ RoiViewModel.cs → ROI 선택 및 상태 표시
- ✅ CropViewModel.cs → ROI 기반 자르기 컨트롤
- ✅ OverlayViewModel.cs → 도형, 라벨 시각화 설정
- ✅ MacroViewModel.cs → 매크로 설정 및 실행 UI
- ✅ PixelInfoViewModel.cs → 마우스 픽셀 정보 표시
- ✅ HistoryViewModel.cs → 상태 이동 인터페이스

---

### **/ViewModels/Commands**

- OpenImageCommand.cs
- SaveImageCommand.cs
- ApplyFilterCommand.cs
- ApplyHistogramCommand.cs
- UndoCommand.cs
- RedoCommand.cs
- SetParameterCommand.cs
- MinimapCommand.cs
- ✅ AddLayerCommand.cs
- ✅ SelectRoiCommand.cs
- ✅ ApplyCropCommand.cs
- ✅ ApplyMacroCommand.cs
- ✅ DrawOverlayCommand.cs
- ✅ InspectPixelCommand.cs
- ✅ NavigateHistoryCommand.cs
- ✅ CopyImageCommand.cs
- ✅ PasteImageCommand.cs