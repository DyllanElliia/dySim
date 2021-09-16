#pragma once
#include <cstdio>

#include "./tensor.hpp"

// getIndex == gi
// It is a template which can help you packing the long parameters to an Vector.
template <typename... Ints>
inline Index gi(Ints... args) {
	int i[] = {(args)...};
	Index a(std::begin(i), std::end(i));
	return a;
}

// printIndex == pi
// This function can transform the index to a string.
std::string pi(const Index index) {
	std::ostringstream out;
	out << "( ";
	for (auto i : index) out << i << " ";
	out << ")";
	return out.str();
}