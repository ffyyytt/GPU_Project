#include "ImageProcessorGPU.hpp"

__device__ float _mod(float x, int modulo)
{
	return int(x) % modulo + (x - int(x));
}

__device__ void _rgb2hsv(unsigned char _r, unsigned char _g, unsigned char _b, float& h, float& s, float& v)
{
    float r = _r / 255., g = _g / 255., b = _b / 255.;
	v = max(max(r, g), b);                            // set v
	float v_min = min(min(r, g), b);
	s = 1. - v_min / v;								    // set s
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

__device__ void _hsv2rgb(float h, float s, float v, unsigned char& _r, unsigned char& _g, unsigned char& _b)
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

__global__ void rgb2hsv(unsigned char* pixels, float* imageHSV, int width, int height)
{
    int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
    while (tid < width*height)
	{
		_rgb2hsv(pixels[3*tid], pixels[3*tid + 1], pixels[3*tid + 2],
				imageHSV[3*tid], imageHSV[3*tid + 1], imageHSV[3*tid + 2]);
		tid += total;
	}
}

__global__ void rgb2hsv_v0(unsigned char* pixels, float* imageHSV, int width, int height)
{
    int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
	int times = ceilf((width*height+0.0)/(total));
	for (int i = 0; i < times; i++)
	{
		if (tid*times + i < width*height)
		{
			_rgb2hsv(pixels[3*(tid*times + i)], pixels[3*(tid*times + i) + 1], pixels[3*(tid*times + i) + 2],
				imageHSV[3*(tid*times + i)], imageHSV[3*(tid*times + i) + 1], imageHSV[3*(tid*times + i) + 2]);
		}
	}
}

__global__ void hsv2rgb(unsigned char* pixels, float* imageHSV, int width, int height)
{
    int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
    while (tid < width*height)
	{
		_hsv2rgb(imageHSV[3*tid], imageHSV[3*tid + 1], imageHSV[3*tid + 2],
				pixels[3*tid], pixels[3*tid + 1], pixels[3*tid + 2]);
		tid += total;
	}
}

__global__ void histogram(float* imageHSV, int* hist, int width, int height, int nbHistogram)
{
    int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
    while (tid < width*height)
	{
		atomicAdd(&hist[(int) round((nbHistogram-1)*imageHSV[3*tid+2])], 1);
		tid += total;
	}
}

__global__ void histogram_mem(float* imageHSV, int* hist, int width, int height, int nbHistogram)
{
	extern __shared__ int histmem[];
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;

	for (int i = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z; i < nbHistogram; i+=blockDim.x*blockDim.y*blockDim.z)
        histmem[i] = 0;
	__syncthreads();

	while (tid < width*height)
	{
		atomicAdd(&histmem[(int) round((nbHistogram-1)*imageHSV[3*tid+2])], 1);
		tid += total;
	}
	__syncthreads();

	for (int i = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z; i < nbHistogram; i+=blockDim.x*blockDim.y*blockDim.z)
        atomicAdd(&hist[i], histmem[i]);
}

__global__ void repart_v1(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	int hist0 = hist[0];

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
	while (tid < nbHistogram)
	{
		float result = 0;
		for (int i = 0; i <= tid; i++)
			result += hist[i];
		cdf[tid] = (result - method*hist0)/(width*height - method*hist0);
		tid += total;
	}
}

__global__ void repart_v2(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	int hist0 = hist[0];

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
	float result = 0;
	while (tid < nbHistogram)
	{
		if (tid < nbHistogram/2)
		{
			result = 0;
			for (int i = 0; i <= tid; i++)
				result += hist[i];
		}
		else
		{
			result = width*height;
			for (int i = nbHistogram - 1; i > tid; i--)
				result -= hist[i];
		}
		cdf[tid] = (result - method*hist0)/(width*height - method*hist0);
		tid += total;
	}
}

__global__ void repart_v2_mem(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	extern __shared__ int histmem[];
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;

	for (int i = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z; i < nbHistogram; i+=blockDim.x*blockDim.y*blockDim.z)
        histmem[i] = hist[i];
	__syncthreads();

	float result = 0;
	while (tid < nbHistogram)
	{
		if (tid < nbHistogram/2)
		{
			result = 0;
			for (int i = 0; i <= tid; i++)
				result += histmem[i];
		}
		else
		{
			result = width*height;
			for (int i = nbHistogram - 1; i > tid; i--)
				result -= histmem[i];
		}
		cdf[tid] = (result - method*histmem[0])/(width*height - method*histmem[0]);
		tid += total;
	}
}

// When i = 0
__global__ void _repart_v3(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	int hist0 = hist[0];

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
	while (tid < nbHistogram)
	{
		if (tid >= method + 1)
			cdf[tid] = (hist[tid] + hist[tid-1]) / (0.0 + width*height - method*hist0);
		else if (tid == method)
			cdf[tid] = hist[tid] / (0.0 + width*height - method*hist0);
		tid += total;
	}
}

// When i > 0
__global__ void _repart_v3(float* src, float* dst, int width, int height, int nbHistogram, int method, int step)
{
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
	while (tid < nbHistogram)
	{
		if (tid >= step)
			dst[tid] = src[tid] + src[tid-step];
		else
			dst[tid] = src[tid];
		tid += total;
	}
}

// Used Kogge-Stone Scan to create CDF.
void repart_v3(dim3 gridDims, dim3 blockDims, int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	float *cdf0, *cdf1;
	cudaMalloc(&cdf0, nbHistogram*sizeof(float));
	cudaMalloc(&cdf1, nbHistogram*sizeof(float));

	int i = 0;
	for (int step = 1; step < nbHistogram; i++, step *=2)
	{
		if (i == 0)
			_repart_v3<<<gridDims, blockDims>>>(hist, cdf0, width, height, nbHistogram, method);
		else if (i%2 == 1)
			_repart_v3<<<gridDims, blockDims>>>(cdf0, cdf1, width, height, nbHistogram, method, step);
		else
			_repart_v3<<<gridDims, blockDims>>>(cdf1, cdf0, width, height, nbHistogram, method, step);
	}

	if (i%2 == 1)
		cudaMemcpy(cdf, cdf0, nbHistogram*sizeof(int), cudaMemcpyDeviceToDevice);
	else
		cudaMemcpy(cdf, cdf1, nbHistogram*sizeof(int), cudaMemcpyDeviceToDevice);
	
	cudaFree(cdf0);
	cudaFree(cdf1);
}

// Reduction i = 0
__global__ void _pre_repart_v4(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	int hist0 = hist[0];

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
	while (tid < nbHistogram)
	{
		if (tid%2 == 1)
		{
			if (method == 0 || tid > 1)
				cdf[tid] = (hist[tid] + hist[tid-1]) / (0.0 + width*height - method*hist0);
			else
				cdf[tid] = hist[tid] / (0.0 + width*height - hist0);
		}
		else
		{
			if (method == 0 || tid > 1)
				cdf[tid] = hist[tid] / (0.0 + width*height - method*hist0);
			else
				cdf[tid] = 0;
		}
		tid += total;
	}
}

// Reduction i > 0
__global__ void _pre_repart_v4(float* cdf, int width, int height, int nbHistogram, int method, int step)
{
	int total = 2*step*(blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z);
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = (x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z + 1)*2*step - 1;
	while (tid < nbHistogram)
	{
		cdf[tid] += cdf[tid - step];
		tid += total;
	}
}

// Post reduction
__global__ void _post_repart_v4(float* cdf, int width, int height, int nbHistogram, int method, int step)
{
	int total = 2*step*(blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z);
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = (x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z + 1)*2*step - 1;
	while (tid + step < nbHistogram)
	{
		cdf[tid + step] += cdf[tid];
		tid += total;
	}
}

__global__ void _repart_v4_mem(int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	extern __shared__ float cdfmem[];
	int total = blockDim.x*blockDim.y*blockDim.z;
    int tid = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;

	int step = 1;
	int hist0 = hist[0];
	for (int i = tid; i < nbHistogram; i+=total)
	{
		if (i == 0 && method == 1)
			continue;
		cdfmem[i] = hist[i] / (width*height - method*hist0 + 0.0);
	}
	__syncthreads();

	// Reduction
	for (; step < nbHistogram; step *=2)
	{
		tid = (tid + 1)*step*2 - 1;
		total = 2*step*total;
		while (tid < nbHistogram)
		{
			cdfmem[tid] += cdfmem[tid - step];
			tid += total;
		}
		tid = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;
		total = blockDim.x*blockDim.y*blockDim.z;
		__syncthreads();
	}

	// Post reduction
	for (step = step/2; step > 0; step /=2)
	{
		tid = (tid + 1)*step*2 - 1;
		total = 2*step*total;
		while (tid + step < nbHistogram)
		{
			cdfmem[tid + step] += cdfmem[tid];
			tid += total;
		}
		tid = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;
		total = blockDim.x*blockDim.y*blockDim.z;
		__syncthreads();
	}

	for (int i = tid; i < nbHistogram; i+=total)
        cdf[i] = cdfmem[i];
}

// Used Brent–Kung Scan to create CDF.
void repart_v4(dim3 gridDims, dim3 blockDims, int* hist, float* cdf, int width, int height, int nbHistogram, int method)
{
	int step = 1;

	if (gridDims.x*gridDims.y*gridDims.z == 1 || blockDims.x*blockDims.y*blockDims.z >= nbHistogram)
	{
		_repart_v4_mem<<<1, blockDims, nbHistogram*sizeof(float)>>>(hist, cdf, width, height, nbHistogram, method);
	}
	else
	{
		// Reduction
		for (; step < nbHistogram; step *=2)
		{
			if (step == 1)
				_pre_repart_v4<<<gridDims, blockDims>>>(hist, cdf, width, height, nbHistogram, method);
			else
				_pre_repart_v4<<<gridDims, blockDims>>>(cdf, width, height, nbHistogram, method, step);
		}

		// Post Reduction
		for (step = step/2; step > 0; step /=2)
		{
			_post_repart_v4<<<gridDims, blockDims>>>(cdf, width, height, nbHistogram, method, step);
		}
	}
}


__global__ void equalization(float* imageHSV, float* cdf, int width, int height, int nbHistogram)
{
	int total = blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockDim.z*gridDim.z;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = x*blockDim.y*gridDim.y*blockDim.z*gridDim.z + y*blockDim.z*gridDim.z + z;
    while (tid < width*height)
	{
		imageHSV[3*tid + 2] = cdf[(int) round((nbHistogram-1)*imageHSV[3*tid + 2])];
		tid += total;
	}
}

void histogramEqualization(dim3 gridDims, dim3 blockDims, Image* image, int nbHistogram, int method)
{
	unsigned char *dev_pixels;
	float *imageHSV;
	int *hist;
	float *cdf;

	// Allocate memory on Device
	cudaMalloc(&dev_pixels, image->_width*image->_height*image->_nbChannels*sizeof(unsigned char));
	cudaMalloc(&imageHSV, image->_width*image->_height*image->_nbChannels*sizeof(float));
	cudaMalloc(&hist, nbHistogram*sizeof(int));
	cudaMalloc(&cdf, nbHistogram*sizeof(float));

	// Set init value to 0
	cudaMemset(hist, 0, nbHistogram*sizeof(int));
	cudaMemset(cdf, 0, nbHistogram*sizeof(int));

	// Copy from image to device
	cudaMemcpy(dev_pixels, image->_pixels, image->_width*image->_height*image->_nbChannels*sizeof(unsigned char), cudaMemcpyHostToDevice);

	// Convert RGB to HSV
	rgb2hsv<<<gridDims, blockDims>>>(dev_pixels, imageHSV, image->_width, image->_height);

	// Histogram
	histogram_mem<<<gridDims, blockDims, nbHistogram*sizeof(int)>>>(imageHSV, hist, image->_width, image->_height, nbHistogram);

	// Repart
	// repart_v2_mem<<<gridDims, blockDims, nbHistogram*sizeof(int)>>>(hist, cdf, image->_width, image->_height, nbHistogram, method);
	repart_v4(gridDims, blockDims, hist, cdf, image->_width, image->_height, nbHistogram, method); // version 3, 4 is call Host fucntion

	// Equalization
	equalization<<<gridDims, blockDims>>>(imageHSV, cdf, image->_width, image->_height, nbHistogram);

	// Convert HSV to RGB
	hsv2rgb<<<gridDims, blockDims>>>(dev_pixels, imageHSV, image->_width, image->_height);

	// Copy back to image
	cudaMemcpy(image->_pixels, dev_pixels, image->_width*image->_height*image->_nbChannels*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	// Free memory
	cudaFree(dev_pixels);
	cudaFree(imageHSV);
	cudaFree(hist);
	cudaFree(cdf);
}