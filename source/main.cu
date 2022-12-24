#include <iostream>
#include "utils/image.hpp"
#include "ImageProcessorCPU.hpp"
#include "ImageProcessorGPU.hpp"
#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

void printUsage() 
{
	std::cerr   << "Usage: " << std::endl
                << " \t -f <F>: <F> image file name" 
                << std::endl << std::endl;
	exit( EXIT_FAILURE );
}

int main( int argc, char **argv )
{	
	char fileName[2048];
	sscanf( argv[2], "%s", &fileName );

	
    Image *imageCPU = new Image();
	imageCPU->load(fileName);
	Image *imageGPU = new Image();
	imageGPU->load(fileName);

	ChronoCPU chrCPU;
	chrCPU.start();
	ImageProcessorCPU* imageProcessorCPU = new ImageProcessorCPU();
	imageProcessorCPU->histogramEqualization(imageCPU, 256, 1);
	chrCPU.stop();
	

	ChronoGPU chrGPU;
	chrGPU.start();
	histogramEqualization(imageGPU, 256, 1);
	chrGPU.stop();

	for (int i = 0; i < imageCPU->_width*imageCPU->_height*imageCPU->_nbChannels; i++)
	{
		if (imageCPU->_pixels[i] != imageGPU->_pixels[i])
			printf("%d %d\n", imageCPU->_pixels[i], imageGPU->_pixels[i]);
	}

	imageCPU->save("image_CPU.png");
	imageGPU->save("image_GPU.png");

	std::cout << chrCPU.elapsedTime() << " " << chrGPU.elapsedTime() << std::endl;
}