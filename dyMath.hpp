#pragma once
#include <cstdio>

#include "./tensor.hpp"

// getIndex == gi
// It is a template which can help you packing the long parameters to an Vector.
template <typename... Ints>
inline Index gi(Ints... args) {
	int i[] = {(args)...};
	Index a(std::begin(i), std::end(i));
	// for (auto ai : a) printf("%d ", ai);
	return a;
}
