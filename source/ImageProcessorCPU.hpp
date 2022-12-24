#ifndef __IMAGEPROCESSORCPU_HPP__
#define __IMAGEPROCESSORCPU_HPP__

#include <iostream>
#include <algorithm>
#include "utils/image.hpp"

class ImageProcessorCPU
{
private:
	float _mod(float x, int modulo);
	void _rgb2hsv(unsigned char _r, unsigned char _g, unsigned char _b, float& h, float& s, float& v);
	void _hsv2rgb(float h, float s, float v, unsigned char& _r, unsigned char& _g, unsigned char& _b);
public:
	ImageProcessorCPU();
	~ImageProcessorCPU();
	void rgb2hsv(unsigned char * pixels, float* imageHSV, int width, int height);
	void hsv2rgb(unsigned char * pixels, float* imageHSV, int width, int height);
	void histogram(float* imageHSV, int* hist, int width, int height, int nbHistogram);
	void repart(int* hist, float* cdf, int width, int height, int nbHistogram, int method);
	void equalization_v1(float* imageHSV, float* cdf, int width, int height, int nbHistogram); 
	void equalization_v2(float* imageHSV, float* cdf, int width, int height, int nbHistogram); // cache

	void histogramEqualization(Image* image, int nbHistogram, int method);
};


#endif // __IMAGEPROCESSORCPU_HPP__