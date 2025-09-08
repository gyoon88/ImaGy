#include "pch.h"
#include "NativeCoreSse.h"
#include <immintrin.h> // For SSE intrinsics
#include <algorithm>   // For std::min

namespace ImaGyNative
{
    namespace SSE
    {
        // Corrected SSE implementation for Average Blur  
        void ApplyAverageBlurSse(void* pixels, int width, int height, int stride, int kernelSize)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride];
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);
            const __m128i mul_div_9 = _mm_set1_epi16(7282); // for x/9 approximation
            __m128i zero = _mm_setzero_si128();

            for (int y = 1; y < height - 1; ++y)
            {
                for (int x = 1; x < vectorizedWidth - 1; x += vectorSize)
                {
                    // Load 9 neighboring 16-pixel blocks
                    __m128i p8_tl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x - 1)));
                    __m128i p8_tc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + x));
                    __m128i p8_tr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x + 1)));
                    __m128i p8_ml = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x - 1)));
                    __m128i p8_mc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + x));
                    __m128i p8_mr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x + 1)));
                    __m128i p8_bl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x - 1)));
                    __m128i p8_bc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + x));
                    __m128i p8_br = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x + 1)));

                    // Process low 8 pixels
                    __m128i sum_lo = _mm_add_epi16(_mm_unpacklo_epi8(p8_tl, zero), _mm_unpacklo_epi8(p8_tc, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_tr, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_ml, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_mc, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_mr, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_bl, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_bc, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_br, zero));
                    __m128i avg_lo = _mm_mulhi_epu16(sum_lo, mul_div_9);

                    // Process high 8 pixels
                    __m128i sum_hi = _mm_add_epi16(_mm_unpackhi_epi8(p8_tl, zero), _mm_unpackhi_epi8(p8_tc, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_tr, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_ml, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_mc, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_mr, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_bl, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_bc, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_br, zero));
                    __m128i avg_hi = _mm_mulhi_epu16(sum_hi, mul_div_9);

                    // Pack and store
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), _mm_packus_epi16(avg_lo, avg_hi));
                }

                // Process remaining pixels
                for (int x = vectorizedWidth; x < width - 1; ++x) {
                    if (x > 0) {
                        int sum = 0;
                        for (int j = -1; j <= 1; ++j) {
                            for (int i = -1; i <= 1; ++i) {
                                sum += sourceBuffer[(y + j) * stride + (x + i)];
                            }
                        }
                        pixelData[y * stride + x] = static_cast<unsigned char>(sum / 9);
                    }
                }
            }
            delete[] sourceBuffer;
        }

        void ApplyGaussianBlurSse(void* pixels, int width, int height, int stride, double sigma, int kernelSize)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride];
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);
            __m128i zero = _mm_setzero_si128();

            const __m128i w1 = _mm_set1_epi16(1);
            const __m128i w2 = _mm_set1_epi16(2);
            const __m128i w4 = _mm_set1_epi16(4);

            for (int y = 1; y < height - 1; ++y)
            {
                for (int x = 1; x < vectorizedWidth - 1; x += vectorSize)
                {
                    __m128i p8_tl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x - 1)));
                    __m128i p8_tc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + x));
                    __m128i p8_tr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x + 1)));
                    __m128i p8_ml = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x - 1)));
                    __m128i p8_mc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + x));
                    __m128i p8_mr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x + 1)));
                    __m128i p8_bl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x - 1)));
                    __m128i p8_bc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + x));
                    __m128i p8_br = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x + 1)));

                    // Process low 8 pixels
                    __m128i sum_lo = _mm_mullo_epi16(_mm_unpacklo_epi8(p8_tl, zero), w1);
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_tc, zero), w2));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_tr, zero), w1));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_ml, zero), w2));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_mc, zero), w4));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_mr, zero), w2));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_bl, zero), w1));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_bc, zero), w2));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_mullo_epi16(_mm_unpacklo_epi8(p8_br, zero), w1));
                    __m128i avg_lo = _mm_srli_epi16(sum_lo, 4);

                    // Process high 8 pixels
                    __m128i sum_hi = _mm_mullo_epi16(_mm_unpackhi_epi8(p8_tl, zero), w1);
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_tc, zero), w2));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_tr, zero), w1));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_ml, zero), w2));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_mc, zero), w4));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_mr, zero), w2));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_bl, zero), w1));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_bc, zero), w2));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_mullo_epi16(_mm_unpackhi_epi8(p8_br, zero), w1));
                    __m128i avg_hi = _mm_srli_epi16(sum_hi, 4);

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), _mm_packus_epi16(avg_lo, avg_hi));
                }

                for (int x = vectorizedWidth; x < width - 1; ++x) {
                    if (x > 0) {
                        int sum = 0;
                        int kernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
                        int k_idx = 0;
                        for (int j = -1; j <= 1; ++j) {
                            for (int i = -1; i <= 1; ++i) {
                                sum += sourceBuffer[(y + j) * stride + (x + i)] * kernel[k_idx++];
                            }
                        }
                        pixelData[y * stride + x] = static_cast<unsigned char>(sum / 16);
                    }
                }
            }
            delete[] sourceBuffer;
        }

        void ApplyDifferentialSse(void* pixels, int width, int height, int stride, unsigned char threshold)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride]; // readonly Buffer !!!
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);

            for (int y = 0; y < height - 1; ++y)
            {
                for (int x = 0; x < vectorizedWidth; x += vectorSize)
                {
                    __m128i p_center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + x));
                    __m128i p_right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x + 1)));
                    __m128i p_down = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + x));

                    // Calculate absolute differences using saturated subtraction
                    __m128i diff_x = _mm_adds_epu8(_mm_subs_epu8(p_right, p_center), _mm_subs_epu8(p_center, p_right));
                    __m128i diff_y = _mm_adds_epu8(_mm_subs_epu8(p_down, p_center), _mm_subs_epu8(p_center, p_down));

                    __m128i sum = _mm_adds_epu8(diff_x, diff_y);

                    // Apply threshold
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), sum);
                }

                for (int x = vectorizedWidth; x < width - 1; ++x)
                {
                    int gradX = (int)sourceBuffer[y * stride + (x + 1)] - (int)sourceBuffer[y * stride + x];
                    int gradY = (int)sourceBuffer[(y + 1) * stride + x] - (int)sourceBuffer[y * stride + x];
                    int val = abs(gradX) + abs(gradY);
                    // Clamp val to 0-255
                    if (val > 255) val = 255;
                    if (val < 0) val = 0; // abs ensures >= 0, but good practice
                    pixelData[y * stride + x] = static_cast<unsigned char>(val);
                }
            }

            delete[] sourceBuffer;
        }

        void ApplySobelSse(void* pixels, int width, int height, int stride, unsigned char threshold)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride];
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);
            __m128i zero = _mm_setzero_si128();

            for (int y = 1; y < height - 1; ++y)
            {
                for (int x = 1; x < vectorizedWidth - 1; x += vectorSize)
                {
                    // Load 8-bit pixel blocks
                    __m128i p8_tl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x - 1)));
                    __m128i p8_tc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + x));
                    __m128i p8_tr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x + 1)));
                    __m128i p8_ml = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x - 1)));
                    __m128i p8_mr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x + 1)));
                    __m128i p8_bl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x - 1)));
                    __m128i p8_bc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + x));
                    __m128i p8_br = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x + 1)));

                    // Process low 8 pixels
                    __m128i p16_tl_lo = _mm_unpacklo_epi8(p8_tl, zero);
                    __m128i p16_tc_lo = _mm_unpacklo_epi8(p8_tc, zero);
                    __m128i p16_tr_lo = _mm_unpacklo_epi8(p8_tr, zero);
                    __m128i p16_ml_lo = _mm_unpacklo_epi8(p8_ml, zero);
                    __m128i p16_mr_lo = _mm_unpacklo_epi8(p8_mr, zero);
                    __m128i p16_bl_lo = _mm_unpacklo_epi8(p8_bl, zero);
                    __m128i p16_bc_lo = _mm_unpacklo_epi8(p8_bc, zero);
                    __m128i p16_br_lo = _mm_unpacklo_epi8(p8_br, zero);

                    __m128i gx_lo = _mm_sub_epi16(p16_tr_lo, p16_tl_lo);
                    gx_lo = _mm_add_epi16(gx_lo, _mm_slli_epi16(p16_mr_lo, 1));
                    gx_lo = _mm_sub_epi16(gx_lo, _mm_slli_epi16(p16_ml_lo, 1));
                    gx_lo = _mm_add_epi16(gx_lo, _mm_sub_epi16(p16_br_lo, p16_bl_lo));

                    __m128i gy_lo = _mm_sub_epi16(p16_bl_lo, p16_tl_lo);
                    gy_lo = _mm_add_epi16(gy_lo, _mm_slli_epi16(p16_bc_lo, 1));
                    gy_lo = _mm_sub_epi16(gy_lo, _mm_slli_epi16(p16_tc_lo, 1));
                    gy_lo = _mm_add_epi16(gy_lo, _mm_sub_epi16(p16_br_lo, p16_tr_lo));

                    // Process high 8 pixels
                    __m128i p16_tl_hi = _mm_unpackhi_epi8(p8_tl, zero);
                    __m128i p16_tc_hi = _mm_unpackhi_epi8(p8_tc, zero);
                    __m128i p16_tr_hi = _mm_unpackhi_epi8(p8_tr, zero);
                    __m128i p16_ml_hi = _mm_unpackhi_epi8(p8_ml, zero);
                    __m128i p16_mr_hi = _mm_unpackhi_epi8(p8_mr, zero);
                    __m128i p16_bl_hi = _mm_unpackhi_epi8(p8_bl, zero);
                    __m128i p16_bc_hi = _mm_unpackhi_epi8(p8_bc, zero);
                    __m128i p16_br_hi = _mm_unpackhi_epi8(p8_br, zero);

                    __m128i gx_hi = _mm_sub_epi16(p16_tr_hi, p16_tl_hi);
                    gx_hi = _mm_add_epi16(gx_hi, _mm_slli_epi16(p16_mr_hi, 1));
                    gx_hi = _mm_sub_epi16(gx_hi, _mm_slli_epi16(p16_ml_hi, 1));
                    gx_hi = _mm_add_epi16(gx_hi, _mm_sub_epi16(p16_br_hi, p16_bl_hi));

                    __m128i gy_hi = _mm_sub_epi16(p16_bl_hi, p16_tl_hi);
                    gy_hi = _mm_add_epi16(gy_hi, _mm_slli_epi16(p16_bc_hi, 1));
                    gy_hi = _mm_sub_epi16(gy_hi, _mm_slli_epi16(p16_tc_hi, 1));
                    gy_hi = _mm_add_epi16(gy_hi, _mm_sub_epi16(p16_br_hi, p16_tr_hi));

                    __m128i sum_lo = _mm_add_epi16(_mm_abs_epi16(gx_lo), _mm_abs_epi16(gy_lo));
                    __m128i sum_hi = _mm_add_epi16(_mm_abs_epi16(gx_hi), _mm_abs_epi16(gy_hi));

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), _mm_packus_epi16(sum_lo, sum_hi));
                }

                // Process remaining pixels
                for (int x = vectorizedWidth; x < width - 1; ++x) {
                    if (x > 0) {
                        int gx = (sourceBuffer[(y - 1) * stride + (x + 1)] + 2 * sourceBuffer[y * stride + (x + 1)] + sourceBuffer[(y + 1) * stride + (x + 1)]) - (sourceBuffer[(y - 1) * stride + (x - 1)] + 2 * sourceBuffer[y * stride + (x - 1)] + sourceBuffer[(y + 1) * stride + (x - 1)]);
                        int gy = (sourceBuffer[(y + 1) * stride + (x - 1)] + 2 * sourceBuffer[(y + 1) * stride + x] + sourceBuffer[(y + 1) * stride + (x + 1)]) - (sourceBuffer[(y - 1) * stride + (x - 1)] + 2 * sourceBuffer[(y - 1) * stride + x] + sourceBuffer[(y - 1) * stride + (x + 1)]);
                        int sum = abs(gx) + abs(gy);
                        if (sum > 255) sum = 255;
                        pixelData[y * stride + x] = static_cast<unsigned char>(sum);
                    }
                }
            }
            delete[] sourceBuffer;
        }

        void ApplyLaplacianSse(void* pixels, int width, int height, int stride, unsigned char threshold)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride];
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);
            __m128i zero = _mm_setzero_si128();
            const __m128i w_neg_8 = _mm_set1_epi16(-8);

            for (int y = 1; y < height - 1; ++y)
            {
                for (int x = 1; x < vectorizedWidth - 1; x += vectorSize)
                {
                    __m128i p8_tl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x - 1)));
                    __m128i p8_tc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + x));
                    __m128i p8_tr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y - 1) * stride + (x + 1)));
                    __m128i p8_ml = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x - 1)));
                    __m128i p8_mc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + x));
                    __m128i p8_mr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + y * stride + (x + 1)));
                    __m128i p8_bl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x - 1)));
                    __m128i p8_bc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + x));
                    __m128i p8_br = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + 1) * stride + (x + 1)));

                    // Process low 8 pixels
                    __m128i sum_lo = _mm_add_epi16(_mm_unpacklo_epi8(p8_tl, zero), _mm_unpacklo_epi8(p8_tc, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_tr, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_ml, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_mr, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_bl, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_bc, zero));
                    sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(p8_br, zero));
                    __m128i center_term_lo = _mm_mullo_epi16(_mm_unpacklo_epi8(p8_mc, zero), w_neg_8);
                    sum_lo = _mm_add_epi16(sum_lo, center_term_lo);
                    sum_lo = _mm_max_epi16(sum_lo, zero); // Clamp to 0

                    // Process high 8 pixels
                    __m128i sum_hi = _mm_add_epi16(_mm_unpackhi_epi8(p8_tl, zero), _mm_unpackhi_epi8(p8_tc, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_tr, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_ml, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_mr, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_bl, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_bc, zero));
                    sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(p8_br, zero));
                    __m128i center_term_hi = _mm_mullo_epi16(_mm_unpackhi_epi8(p8_mc, zero), w_neg_8);
                    sum_hi = _mm_add_epi16(sum_hi, center_term_hi);
                    sum_hi = _mm_max_epi16(sum_hi, zero); // Clamp to 0

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), _mm_packus_epi16(sum_lo, sum_hi));
                }

                for (int x = vectorizedWidth; x < width - 1; ++x) {
                    if (x > 0) {
                        int sum = 0;
                        int kernel[9] = { 1, 1, 1, 1, -8, 1, 1, 1, 1 };
                        int k_idx = 0;
                        for (int j = -1; j <= 1; ++j) {
                            for (int i = -1; i <= 1; ++i) {
                                sum += (int)sourceBuffer[(y + j) * stride + (x + i)] * kernel[k_idx++];
                            }
                        }
                        if (sum < 0) sum = 0;
                        if (sum > 255) sum = 255;
                        pixelData[y * stride + x] = static_cast<unsigned char>(sum);
                    }
                }
            }
            delete[] sourceBuffer;
        }

        void ApplyDilationSse(void* pixels, int width, int height, int stride, unsigned char threshold)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride];
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);

            for (int y = 1; y < height - 1; ++y)
            {
                for (int x = 1; x < vectorizedWidth - 1; x += vectorSize)
                {
                    __m128i max_val = _mm_setzero_si128();
                    for (int j = -1; j <= 1; ++j) {
                        for (int i = -1; i <= 1; ++i) {
                            __m128i neighbor = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + j) * stride + (x + i)));
                            max_val = _mm_max_epu8(max_val, neighbor);
                        }
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), max_val);
                }

                for (int x = vectorizedWidth; x < width - 1; ++x) {
                    if (x > 0) {
                        unsigned char maxVal = 0;
                        for (int j = -1; j <= 1; ++j) {
                            for (int i = -1; i <= 1; ++i) {
                                unsigned char neighborPixel = sourceBuffer[(y + j) * stride + (x + i)];
                                if (neighborPixel > maxVal) {
                                    maxVal = neighborPixel;
                                }
                            }
                        }
                        pixelData[y * stride + x] = maxVal;
                    }
                }
            }
            delete[] sourceBuffer;
        }

        void ApplyErosionSse(void* pixels, int width, int height, int stride, unsigned char threshold)
        {
            unsigned char* pixelData = static_cast<unsigned char*>(pixels);
            unsigned char* sourceBuffer = new unsigned char[height * stride];
            memcpy(sourceBuffer, pixelData, height * stride);

            const int vectorSize = 16;
            int vectorizedWidth = width - (width % vectorSize);

            for (int y = 1; y < height - 1; ++y)
            {
                for (int x = 1; x < vectorizedWidth - 1; x += vectorSize)
                {
                    __m128i min_val = _mm_set1_epi8(-1); // Initialize with 255
                    for (int j = -1; j <= 1; ++j) {
                        for (int i = -1; i <= 1; ++i) {
                            __m128i neighbor = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sourceBuffer + (y + j) * stride + (x + i)));
                            min_val = _mm_min_epu8(min_val, neighbor);
                        }
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(pixelData + y * stride + x), min_val);
                }

                for (int x = vectorizedWidth; x < width - 1; ++x) {
                    if (x > 0) {
                        unsigned char minVal = 255;
                        for (int j = -1; j <= 1; ++j) {
                            for (int i = -1; i <= 1; ++i) {
                                unsigned char neighborPixel = sourceBuffer[(y + j) * stride + (x + i)];
                                if (neighborPixel < minVal) {
                                    minVal = neighborPixel;
                                }
                            }
                        }
                        pixelData[y * stride + x] = minVal;
                    }
                }
            }
            delete[] sourceBuffer;
        }
    }
}
