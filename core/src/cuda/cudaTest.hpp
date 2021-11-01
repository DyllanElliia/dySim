#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void Decrease2(int *a, int *b, int *c);
void addWithCuda2(int *c, int *a, int *b);