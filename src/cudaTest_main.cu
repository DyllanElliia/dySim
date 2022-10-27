/*
 * @Author: DyllanElliia
 * @Date: 2021-09-22 14:21:25
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-11-09 11:52:50
 * @Description: How to use Tensor.
 */

// #include <algorithm>
#include <dyMath.hpp>


template <typename T>
__global__ void foo(T in) { printf("-> value = %d\n", in()); }
struct S1_t {
    int xxx;
    __host__ __device__ S1_t(void) : xxx(10) { };
    void doit(void) {
        // note the "*this" capture specification
        auto lam1 = [=] __device__{
            glm::vec3 v={1,2,3};
            int i = blockIdx.x+1;
            int j = threadIdx.x+1;
            if constexpr(1>0)
                printf("test success\n");
            auto autoTest=1;
            std::cout<<"?";
            printf("coord: %d---%f, %d\n",i*100+j,v[0],autoTest);

            return i*100+j;
        };
        // Kernel launch succeeds
        foo <<<2, 4 >>>(lam1);
            cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error: %s.\n",cudaGetErrorString(cudaerr));
               
    } 
};
int main(void) {
    printf("run\n");
    S1_t s1;
    s1.doit();
    printf("end\n");
}