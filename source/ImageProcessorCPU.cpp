#include "ImageProcessorCPU.hpp"

float ImageProcessorCPU::_mod(float x, int modulo)
{
	return int(x) % modulo + (x - int(x));
}

void ImageProcessorCPU::_rgb2hsv(unsigned char _r, unsigned char _g, unsigned char _b, float & h, float & s, float & v)
{
	float r = _r / 255., g = _g / 255., b = _b / 255.;
	v = std::max(std::max(r, g), b);                            // set v
	float v_min = std::min(std::min(r, g), b);
	s = 1. - v_min / v;								           // set s
	if (v == v_min)
		h = 0;
	else if (v == r)
		h = 60 * (g - b) / (v - v_min);
	else if (v == g)
		h = 120 + 60 * (b - r) / (v - v_min);
	else if (v == b)
		h = 240 + 60 * (r - g) / (v - v_min);
	if (h < 0)
		h += 360;
}

void ImageProcessorCPU::_hsv2rgb(float h, float s, float v, unsigned char& _r, unsigned char& _g, unsigned char& _b)
{
	float r, g, b;
	float c = v * s;
	float m = v - c;
	h = h / 60;
	float x = c * (1 - abs(_mod(h, 2) - 1));
	if (h < 1)
	{
		r = c;
		g = x;
		b = 0;
	}
	else if (h < 2)
	{
		r = x;
		g = c;
		b = 0;
	}
	else if (h < 3)
	{
		r = 0;
		g = c;
		b = x;
	}
	else if (h < 4)
	{
		r = 0;
		g = x;
		b = c;
	}
	else if (h < 5)
	{
		r = x;
		g = 0;
		b = c;
	}
	else
	{
		r = c;
		g = 0;
		b = x;
	}
	_r = round((r + m) * 255);
	_g = round((g + m) * 255);
	_b = round((b + m) * 255);
}

ImageProcessorCPU::ImageProcessorCPU()
{
}

ImageProcessorCPU::~ImageProcessorCPU()
{
}

void ImageProcessorCPU::rgb2hsv(unsigned char* pixels, float* imageHSV, int width, int height)
{
	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			_rgb2hsv(pixels[3 * (x*width + y)], pixels[3 * (x*width + y) + 1], pixels[3 * (x*width + y) + 2],
				imageHSV[3 * (x*width + y)], imageHSV[3 * (x*width + y) + 1], imageHSV[3 * (x*width + y) + 2]);
		}
	}
}

void ImageProcessorCPU::hsv2rgb(unsigned char * pixels, float* imageHSV, int width, int height)
{
	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			_hsv2rgb(imageHSV[3 * (x*width + y)], imageHSV[3 * (x*width + y) + 1], imageHSV[3 * (x*width + y) + 2],
				pixels[3 * (x*width + y)], pixels[3 * (x*width + y) + 1], pixels[3 * (x*width + y) + 2]);
		}
	}
}

void ImageProcessorCPU::histogram(float* imageHSV, int* hist, int width, int height, int nbHistogram)
{
	memset(hist, 0, nbHistogram * sizeof(float));
	for (int i = 0; i < width*height; i++)
		hist[(int) round(imageHSV[3 * i + 2] * (nbHistogram - 1))] += 1;
}

void ImageProcessorCPU::repart(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	cdf[0] = (1-method)*hist[0];
	// if (medthod == 0)
	// 	cdf[0] = hist[0];
	// else
	// 	cdf[0] = 0;
	for (int i = 1; i < nbHistogram; i++)
		cdf[i] = cdf[i-1] + hist[i];
}

void ImageProcessorCPU::equalization_v1(float* imageHSV, float* cdf, int width, int height, int nbHistogram)
{
	for (int i = 0; i < width*height; i++)
		imageHSV[3*i + 2] = cdf[(int) round(imageHSV[3 * i + 2] * (nbHistogram - 1))] / cdf[nbHistogram-1];
}

void ImageProcessorCPU::equalization_v2(float* imageHSV, float* cdf, int width, int height, int nbHistogram)
{
	for (int i = 0; i < nbHistogram; i++)
		cdf[i] = cdf[i]/cdf[nbHistogram-1];

	for (int i = 0; i < width*height; i++)
		imageHSV[3 * i + 2] = cdf[(int) round(imageHSV[3 * i + 2] * (nbHistogram - 1))];
}

void ImageProcessorCPU::histogramEqualization(Image* image, int nbHistogram, int method)
{
	float *imageHSV = (float*)malloc(image->_width*image->_height*image->_nbChannels * sizeof(float));
	int *hist = (int*)malloc(nbHistogram*sizeof(int));
	float *cdf = (float*)malloc(nbHistogram*sizeof(float));

	rgb2hsv(image->_pixels, imageHSV, image->_width, image->_height);
	histogram(imageHSV, hist, image->_width, image->_height, nbHistogram);
	repart(hist, cdf, image->_width, image->_height, nbHistogram, method);
	equalization_v2(imageHSV, cdf, image->_width, image->_height, nbHistogram);
	
	hsv2rgb(image->_pixels, imageHSV, image->_width, image->_height);

	delete[] imageHSV;
	delete[] hist;
	delete[] cdf;
}