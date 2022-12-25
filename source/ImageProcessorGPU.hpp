#ifndef __IMAGEPROCESSORGPU_HPP__
#define __IMAGEPROCESSORGPU_HPP__

#include "stdio.h"
#include "utils/image.hpp"
#include "utils/commonCUDA.hpp"

__device__ float _mod(float x, int modulo);
__device__ void _rgb2hsv(unsigned char _r, unsigned char _g, unsigned char _b, float& h, float& s, float& v);
__device__ void _hsv2rgb(float h, float s, float v, unsigned char& _r, unsigned char& _g, unsigned char& _b);

__global__ void rgb2hsv(unsigned char* pixels, float* imageHSV, int width, int height);
__global__ void hsv2rgb(unsigned char* pixels, float* imageHSV, int width, int height);
__global__ void histogram(float* imageHSV, int* hist, int width, int height, int nbHistogram);
__global__ void histogram_mem(float* imageHSV, int* hist, int width, int height, int nbHistogram);
__global__ void repart_v1(int* hist, float* cdf, int width, int height, int nbHistogram, int method);
__global__ void repart_v2(int* hist, float* cdf, int width, int height, int nbHistogram, int method);
void repart_v3(dim3 gridDims, dim3 blockDims, int* hist, float* cdf, int width, int height, int nbHistogram, int method);

__global__ void equalization(float* imageHSV, float* cdf, int width, int height, int nbHistogram);

void histogramEqualization(dim3 gridDims, dim3 blockDims, Image* image, int nbHistogram, int method);

#endif // __IMAGEPROCESSORGPU_HPP__