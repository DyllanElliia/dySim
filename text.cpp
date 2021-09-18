#include "./dyMath.hpp"
#include "./dyPicture.hpp"

int main() {
	std::vector<short> v{0, 2, 3};
	pixel<3> a(v);
	pixel<3> b({4, 2, 3});
	pixel<3> c = a / b;
	std::cout << c << std::endl;
	/* 	tensor<int> a(
		gi(3, 2, 3), []() -> std::vector<int> {
            int x=3,y=4;
            std::vector<int> v(3*2*3,1);
            v[x] = y;
            return v; });
	tensor<int> b(
		gi(3, 2), [](const Index& shape) -> std::vector<int> {
	        std::vector<int> v(3*2*3,2);
	        v[i2ti(gi(2,1), shape)] = 9;
	        return v; });

	matrix<int> ma(gi(4, 3), []() -> std::vector<int> {
		std::vector<int> v(4 * 3);
		int count = 0;
		for (auto& i : v) i = count++;
		return v;
	});

	std::vector<std::vector<int>> v{
		{1, 2, 3, 4},
		{1, 2, 3, 4},
		{1, 2, 3, 4}};
	matrix<int> mb(v);

	ma[gi(1, 1)] = 0;
	ma.show();
	mb.show();
	std::cout << "operator test!" << std::endl;
	matrix<int> mc = ma * mb;
	mc.show();
	(mc * 3).show();
	(mc / 2).show();
	(mc + mc + mc - mc / 2).show();
	((mc + mc + mc - mc / 2) * mc).show(); */

	// const size_t im = 4, jm = 4, km = 3;
	// for (size_t i = 0; i < im; ++i)
	// 	for (size_t k = 0; k < km; ++k)
	// 		for (size_t j = 0; j < jm; ++j)
	// 			mc[gi(i, j)] += ma[gi(i, k)] * mb[gi(k, j)];
	// mc.show();
	// std::cout << pi(gi(3, 2, 1) * gi(2, 2, 2)) << std::endl;
	// a[gi(1, 0, 0)] = 2;
	// a[gi(2, 1, 1)] = 3;
	// std::cout << "run" << std::endl;
	// a.show();
	// b.show();
	// tensor<int> c = a / 2;
	// std::cout << c << std::endl
	// 		  << pi(c.shape()) << std::endl;
	// std::cout << "runPlus" << std::endl;
	return 0;
}
