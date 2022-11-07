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
      std::array<Real,2> asdf={1.11,2.22};
      dym::Vector3 lkfj{3.33,2.22,1.11};
      glm::vec3 vnxm{1,1,1};
      printf("%d, %d\n",sizeof(lkfj),sizeof(vnxm));
        // note the "*this" capture specification
        const auto lam1 = [=] __device__  (){
            dym::Vector3 fsfd(vnxm);
            dym::Vector3 tryF([=](int i){return (Real)i;});
            Real f=1e-3;
            auto printv=[&]_DYM_GENERAL_(Real i){printf("%f = %f\n",i,fsfd[0]);};
            auto a111=lkfj;
            auto a11i=lkfj.cast<int>();

            // fsfd+=1.;
            glm::vec3 v={1,2,3};
            int i = blockIdx.x+1;
            int j = threadIdx.x+1;
            if constexpr(1>0)
                printf("test success\n");
                            printv(tryF[2]);
            auto autoTest=1; 
            // std::cout<<"?";
            tryF*=tryF+fsfd;
            printf("1 %f, %f, %f\n",fsfd[0],fsfd[1],fsfd[2]);
            printf("2 %f, %f, %f\n",tryF[0],tryF[1],tryF[2]);
            printf("3 %f, %f, %f\n", tryF.dot(a111),tryF.length(),tryF.normalize()[1]);
            printf("4 %d\n",a11i[0]);
            printf("coord: %d---%f, %d, %f, %f\n",i*100+j,v[0],autoTest,asdf[0],asdf[1]);
            
            // printf("str: %s\n",std::string("is string").c_str());
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