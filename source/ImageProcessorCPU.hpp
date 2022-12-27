#ifndef __IMAGEPROCESSORCPU_HPP__
#define __IMAGEPROCESSORCPU_HPP__

#include <cmath>
#include <iostream>
#include <algorithm>
#include "utils/image.hpp"

class ImageProcessorCPU
{
private:
	// modulo a float
	float _mod(float x, int modulo);

	// Convert one pixel RGB to HSV
	void _rgb2hsv(unsigned char _r, unsigned char _g, unsigned char _b, float& h, float& s, float& v);

	// Convert one pixel HSV to RGB
	void _hsv2rgb(float h, float s, float v, unsigned char& _r, unsigned char& _g, unsigned char& _b);
public:
	ImageProcessorCPU();
	~ImageProcessorCPU();

	// Convert one image RGB to HSV
	void rgb2hsv(unsigned char * pixels, float* imageHSV, int width, int height);

	// Convert one image HSV to RGB
	void hsv2rgb(unsigned char * pixels, float* imageHSV, int width, int height);
	void histogram(float* imageHSV, int* hist, int width, int height, int nbHistogram);
	void repart(int* hist, float* cdf, int width, int height, int nbHistogram, int method);
	void equalization_v1(float* imageHSV, float* cdf, int width, int height, int nbHistogram); 
	void equalization_v2(float* imageHSV, float* cdf, int width, int height, int nbHistogram); // divide first

	void histogramEqualization(Image* image, int nbHistogram, int method);
};


#endif // __IMAGEPROCESSORCPU_HPP__