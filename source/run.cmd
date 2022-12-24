nvcc -o main main.cu .\ImageProcessorCPU.cpp .\ImageProcessorGPU.cu .\utils\image.cpp .\utils\chronoCPU.cpp .\utils\chronoGPU.cu
main -f .\images\Chateau.png
main -f .\images\palais_garnier.jpg