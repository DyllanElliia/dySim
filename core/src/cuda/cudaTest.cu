#include"./cudaTest.hpp"

__global__ void Decrease2(int *a, int *b, int *c) { *c = *a - *b; }
void addWithCuda2(int *c, int *a, int *b) {
  int *dev_c = 0;
  int *dev_a = 0;
  int *dev_b = 0;
  // 3.请求CUDA设备的内存（显存），执行CUDA函数
  cudaMalloc((void **)&dev_c, sizeof(int));
  cudaMalloc((void **)&dev_a, sizeof(int));
  cudaMalloc((void **)&dev_b, sizeof(int));

  // 4.从主机复制数据到设备上
  cudaMemcpy(dev_a, a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof(int), cudaMemcpyHostToDevice);

  Decrease2<<<1, 1>>>(dev_a, dev_b, dev_c);

  // 5.等待设备所有线程任务执行完毕
  cudaDeviceSynchronize();

  // 6.数据复制到主机，释放占用空间
  cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaDeviceReset();         // 7.重置CUDA设备释放程序占用的资源
}