#include <iostream>
#include "utils/image.hpp"
#include "ImageProcessorCPU.hpp"
#include "ImageProcessorGPU.hpp"
#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

char *inputFile = "images/Chateau.png", *outputFile="image.png";
int nbHistogram = 256, method = 0, hardware = 0;
int gridDimx = 1024, gridDimy = 1, gridDimz = 1;
int blockDimx = 1024, blockDimy = 1, blockDimz = 1;

void printUsage() 
{
	fprintf(stderr, "There are maximum 10 parameters:\n");
	fprintf(stderr, "\t1. Input image file, default: %s\n", inputFile);
	fprintf(stderr, "\t2. Output image file, default: %s\n", outputFile);
	fprintf(stderr, "\t3. Scaler (0=MaxAbsScaler, 1=MinMaxScaler), default: %d\n", method);
	fprintf(stderr, "\t4. nbHistogram, default: %d\n", nbHistogram);
	fprintf(stderr, "\t5. CPU/GPU (0=CPU, 1=GPU), default: %d\n", hardware);
	fprintf(stderr, "\t6. GPU gridDim.x, default: %d\n", gridDimx);
	fprintf(stderr, "\t7. GPU gridDim.y, default: %d\n", gridDimy);
	fprintf(stderr, "\t8. GPU gridDim.z, default: %d\n", gridDimz);
	fprintf(stderr, "\t9. GPU blockDim.x, default: %d\n", blockDimx);
	fprintf(stderr, "\t10. GPU blockDim.y, default: %d\n", blockDimy);
	fprintf(stderr, "\t11. GPU blockDim.z, default: %d\n", blockDimz);
	exit(0);
	exit( EXIT_FAILURE );
}

int main( int argc, char **argv )
{
	if (!strcmp(argv[argc-1], "--help"))
		printUsage();

	if (argc>=2) inputFile = argv[1];
    if (argc>=3) outputFile = argv[2];
    if (argc>=4) method = atoi(argv[3]);
	if (argc>=5) nbHistogram = atoi(argv[4]);
    if (argc>=6) hardware = atoi(argv[5]);
    if (argc>=7) gridDimx = atoi(argv[6]);
    if (argc>=8) gridDimy = atoi(argv[7]);
    if (argc>=9) gridDimz = atoi(argv[8]);
    if (argc>=10) blockDimx = atoi(argv[9]);
    if (argc>=11) blockDimy = atoi(argv[10]);
	if (argc>=12) blockDimz = atoi(argv[11]);
    if (argc >13) printUsage();

	if (hardware > 1)
	{
		hardware = 0;
		fprintf(stderr, "Only supoort 2 types of hardware (0=CPU, 1=GPU). Has been changed to %d", hardware);
	}
	if (method > 1)
	{
		method = 0;
		fprintf(stderr, "Only supoort 2 method (0=MaxAbsScaler, 1=MinMaxScaler). Has been changed to %d", method);
	}

	dim3 gridDims(gridDimx, gridDimy, gridDimz);
	dim3 blockDims(blockDimx, blockDimy, blockDimz);

	Image *image = new Image();
	image->load(inputFile);

	ChronoCPU chrCPU;
	ChronoGPU chrGPU;
	ImageProcessorCPU* imageProcessorCPU = new ImageProcessorCPU();

	switch (hardware)
	{
	case 0:
		chrCPU.start();
		imageProcessorCPU->histogramEqualization(image, nbHistogram, method);
		chrCPU.stop();
		std::cout << chrCPU.elapsedTime()<< std::endl;
		break;
	default:
		chrGPU.start();
		histogramEqualization(gridDims, blockDims, image, nbHistogram, method);
		chrGPU.stop();
		std::cout<< chrGPU.elapsedTime() << std::endl;
		break;
	}

	image->save(outputFile);
}