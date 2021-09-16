#include "./dyMath.hpp"

int main() {
	tensor<int> a(
		gi(3, 2, 3), []() -> std::vector<int> {
            int x=3,y=4;
            std::vector<int> v(3*2*3,1);
            v[x] = y;
            return v; });
	tensor<int> b(
		gi(3, 2, 3), []() -> std::vector<int> {
            int x=3,y=4;
            std::vector<int> v(3*2*3,2);
            v[x] = y;
            return v; });
	a[gi(1, 0, 0)] = 2;
	a[gi(2, 1, 1)] = 3;
	// std::cout << "run" << std::endl;
	a.show();
	// b.show();
	tensor<int> c = a.cut(gi(0, 0, 0), gi(3, 2, 1));
	std::cout << c << std::endl
			  << pi(c.shape()) << std::endl;
	std::cout << "runPlus" << std::endl;
	return 0;
}
