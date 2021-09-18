#include <array>

#include "./dyMath.hpp"

template <const int color_size>
struct pixel {
	// #define color_size 3
	std::vector<short> color;
	pixel(std::vector<short> color_) {
		color.assign(color_.begin(), color_.begin() + color_size);
	}
	pixel() { color.reserve(color_size); }
	pixel(pixel& p) : color(p.color) {}
	pixel(pixel&& p) : color(p.color) {}

	template <class tranFun, class T>
	static T computer(const T& first, const T& second) {
		T result;
		std::transform(first.color.begin(), first.color.end(), second.color.begin(), std::back_inserter(result.color), tranFun());
		return result;
	}

	void operator=(const pixel& p) { color = p.color; }

	friend std::ostream& operator<<(std::ostream& output, pixel& p) {
		output << "(";
		for (auto i : p.color) output << " " << i;
		output << " )";
		return output;
	}

	friend pixel
	operator+(const pixel& first, const pixel& second) {
		return computer<std::plus<short> >(first, second);
	}
	friend pixel operator+(const int& first, const pixel& second) {
		pixel<color_size> result(second);
		for (auto& i : result) i = first + i;
		return result;
	}
	friend pixel operator+(const pixel& first, const int& second) {
		pixel<color_size> result(first);
		for (auto& i : result) i = i + second;
		return result;
	}

	friend pixel operator-(const pixel& first, const pixel& second) {
		return computer<std::minus<short> >(first, second);
	}
	friend pixel operator-(const int& first, const pixel& second) {
		pixel<color_size> result(second);
		for (auto& i : result) i = first - i;
		return result;
	}
	friend pixel operator-(const pixel& first, const int& second) {
		pixel<color_size> result(first);
		for (auto& i : result) i = i - second;
		return result;
	}

	friend pixel operator*(const pixel& first, const pixel& second) {
		return computer<std::multiplies<short> >(first, second);
	}
	friend pixel operator*(const int& first, const pixel& second) {
		pixel<color_size> result(second);
		for (auto& i : result) i = first * i;
		return result;
	}
	friend pixel operator*(const pixel& first, const int& second) {
		pixel<color_size> result(first);
		for (auto& i : result) i = i * second;
		return result;
	}

	friend pixel operator/(const pixel& first, const pixel& second) {
		return computer<std::divides<short> >(first, second);
	}
	friend pixel operator/(const int& first, const pixel& second) {
		pixel<color_size> result(second);
		for (auto& i : result) i = first / i;
		return result;
	}
	friend pixel operator/(const pixel& first, const int& second) {
		pixel<color_size> result(first);
		for (auto& i : result) i = i / second;
		return result;
	}

	short& operator[](const int& index_) { return color[index_ % color_size]; }
};