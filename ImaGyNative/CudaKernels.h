#pragma once

// CUDA Binarization Launcher
// Returns true on success, false on failure.
bool LaunchBinarizationKernel(
    unsigned char* pixels,
    int width,
    int height,
    int stride,
    int threshold
);