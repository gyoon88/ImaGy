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

**4. C++ Native DLL Loading Issue Resolution:**
- The primary issue of `ImaGyWrapper.dll` failing to find `ImaGyNative.dll` at runtime was diagnosed.
- It was determined that `ImaGyNative.dll` was being built successfully in its own output directory (`$(SolutionDir)x64\$(Configuration)\`) but was not being copied to the main `ImaGy` application's output directory (`ImaGy\bin\x64\Debug\net8.0-windows10.0.26100.0\`).
- This was resolved by explicitly adding `ImaGyNative.dll` as a content item to `ImaGy.csproj` with `CopyToOutputDirectory` set to `PreserveNewest`, ensuring it's copied to the final application output.

--- 

### **NEXT STEPS**

The next task is to implement other image processing functionalities in C++ within the `ImaGyNative` project, leveraging the established C# to C++ interoperability.
