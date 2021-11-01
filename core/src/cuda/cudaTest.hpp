/*
 * @Author: DyllanElliia
 * @Date: 2021-11-01 20:11:36
 * @LastEditTime: 2021-11-01 20:20:58
 * @LastEditors: DyllanElliia
 * @Description: 
 */
#include <cuda_runtime.h>
// #include <cublas.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void Decrease2(int *a, int *b, int *c);
void addWithCuda2(int *c, int *a, int *b);