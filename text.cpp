#include "./dyMath.hpp"

int main() {
	tensor<int> a(gi(3, 2, 3, 2), 0);
	a[gi(1, 0, 0, 1)] = 2;
	a[gi(2, 1, 1, 0)] = 3;
	a.show("|   ");
	return 0;
}
