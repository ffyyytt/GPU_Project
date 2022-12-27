#ifndef __IMAGEPROCESSORGPU_HPP__
#define __IMAGEPROCESSORGPU_HPP__

#include "stdio.h"
#include "utils/image.hpp"
#include "utils/commonCUDA.hpp"

// modulo a float
__device__ float _mod(float x, int modulo);

// Convert one pixel RGB to HSV
__device__ void _rgb2hsv(unsigned char _r, unsigned char _g, unsigned char _b, float& h, float& s, float& v);

// Convert one pixel HSV to RGB
__device__ void _hsv2rgb(float h, float s, float v, unsigned char& _r, unsigned char& _g, unsigned char& _b);

// Convert one image RGB to HSV
__global__ void rgb2hsv(unsigned char* pixels, float* imageHSV, int width, int height);

// Convert one image HSV to RGB
__global__ void hsv2rgb(unsigned char* pixels, float* imageHSV, int width, int height);

// Normal histogram
__global__ void histogram(float* imageHSV, int* hist, int width, int height, int nbHistogram);

// Histogram shared memory
__global__ void histogram_mem(float* imageHSV, int* hist, int width, int height, int nbHistogram);

// Baseline
__global__ void repart_v1(int* hist, float* cdf, int width, int height, int nbHistogram, int method);

// Baseline optimize
__global__ void repart_v2(int* hist, float* cdf, int width, int height, int nbHistogram, int method);

// Used Kogge-Stone Scan to create CDF.
void repart_v3(dim3 gridDims, dim3 blockDims, int* hist, float* cdf, int width, int height, int nbHistogram, int method);

// Used Brent–Kung Scan to create CDF.
void repart_v4(dim3 gridDims, dim3 blockDims, int* hist, float* cdf, int width, int height, int nbHistogram, int method);

__global__ void equalization(float* imageHSV, float* cdf, int width, int height, int nbHistogram);

void histogramEqualization(dim3 gridDims, dim3 blockDims, Image* image, int nbHistogram, int method);

#endif // __IMAGEPROCESSORGPU_HPP__