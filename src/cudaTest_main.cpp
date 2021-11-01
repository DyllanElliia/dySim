/*
 * @Author: DyllanElliia
 * @Date: 2021-11-01 20:04:41
 * @LastEditTime: 2021-11-01 20:04:41
 * @LastEditors: DyllanElliia
 * @Description: 
 */
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <stdio.h>

#include <src/cuda/cudaTest.hpp>


int main(void) {
  int c;
  int a, b;
  c = 10;
  a = 30;
  b = 15;
  addWithCuda2(&c, &a, &b);  // 2.传入参数变量（地址）
  printf("Value is %d!", c); // 8.主机上打印显示数据
  return 0;
}