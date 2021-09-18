#include "./dyMath.hpp"

struct pix_gray {
	uint8_t color;
	pix_gray(uint8_t color) : color(color) {}
	pix_gray() {}
	pix_gray(pix_gray& p) : color(p.color) {}
	pix_gray(pix_gray&& p) : color(p.color) {}

	friend pix_gray operator+(const pix_gray& first, const pix_gray& second) {
		return pix_gray(first.color + second.color);
	}
	friend pix_gray operator-(const pix_gray& first, const pix_gray& second) {
		return pix_gray(first.color - second.color);
	}
	friend pix_gray operator*(const pix_gray& first, const pix_gray& second) {
		return pix_gray(first.color * second.color);
	}
	friend pix_gray operator/(const pix_gray& first, const pix_gray& second) {
		return pix_gray(first.color / second.color);
	}
};