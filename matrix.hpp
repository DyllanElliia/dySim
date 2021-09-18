#pragma once
#include "./tensor.hpp"

template <class T>
class matrix : public tensor<T> {
protected:
	using ValueType = T;
	using ValueUse = T&;
	using ValuePtr = T*;

	using tensor<ValueType>::a;
	using tensor<ValueType>::tsShape;
	using tensor<ValueType>::tsShapeSuffix;
	using tensor<ValueType>::updateSuffix;
	// using tensor<ValueType>::operator+;
	// template <class tranFun, class comValue>
	// using computer = tensor<ValueType>::computer<tranFun, comValue>;

	inline void shapeCheck(const Index& shape) {
		// std::cout << "run sc!" << std::endl;
		// std::cout << shape.size() << std::endl;
		// if (shape.size() != 2)std::cout << "asdf" << std::endl;
		try {
			if (shape.size() != 2)
				throw "matrix shape must be 2-dimensional!";
		} catch (const char* str) {
			std::cerr << str << '\n';
			exit(EXIT_FAILURE);
		}
	}

	template <class tranFun>
	static matrix computer(const matrix& first, const matrix& second) {
		matrix result;
		result.tsShape = first.tsShape;
		result.updateSuffix();
		result.a.reserve(first.a.size());
		std::transform(first.a.begin(), first.a.end(), second.a.begin(), std::back_inserter(result.a), tranFun());
		return result;
	}

public:
	matrix(const Index& shape, const ValueType defaultValue = 0) : tensor<ValueType>(shape, defaultValue) {
		shapeCheck(shape);
	}
	matrix(const Index& shape, std::function<std::vector<ValueType>()> creatFun) : tensor<ValueType>(shape, creatFun) {
		shapeCheck(shape);
	}
	matrix(const Index& shape, std::function<std::vector<ValueType>(const Index& shape)> creatFun) : tensor<ValueType>(shape, creatFun) {
		shapeCheck(shape);
	}
	matrix(matrix<ValueType>&& ts) : tensor<ValueType>() {
		shapeCheck(ts.tsShape);
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
	matrix(matrix<ValueType>& ts) : tensor<ValueType>() {
		shapeCheck(ts.tsShape);
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
	matrix(tensor<ValueType>&& ts) : tensor<ValueType>() {
		shapeCheck(ts.tsShape);
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
	matrix(tensor<ValueType>& ts) : tensor<ValueType>() {
		shapeCheck(ts.tsShape);
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
	matrix(std::vector<std::vector<ValueType>>& v) : tensor<ValueType>() {
		tsShape.push_back(v.size());
		int my = 1e7;
		for (auto& l : v) my = std::min(my, (int)l.size());
		tsShape.push_back(my);
		for (auto& l : v) a.insert(a.end(), l.begin(), l.begin() + my);
		updateSuffix();
	}
	matrix() : tensor<ValueType>() {}
	~matrix() {}

	friend matrix operator+(const matrix& first, const matrix& second) {
		return computer<std::plus<ValueType>>(first, second);
	}

	/* 	matrix operator+(const ValueType& second) {
		matrix result(*this);
		for (auto& i : result.a) i = i + second;
		return result;
	} */

	friend matrix operator+(const ValueType& first, const matrix& second) {
		matrix result(second);
		for (auto& i : result) i = first + i;
		return result;
	}

	friend matrix operator+(const matrix& first, const ValueType& second) {
		matrix result(first);
		for (auto& i : result) i = i + second;
		return result;
	}

	friend matrix operator-(const matrix& first, const matrix& second) {
		return computer<std::minus<ValueType>>(first, second);
	}

	friend matrix operator-(const ValueType& first, const matrix& second) {
		matrix result(second);
		for (auto& i : result) i = first - i;
		return result;
	}

	friend matrix operator-(const matrix& first, const ValueType& second) {
		matrix result(first);
		for (auto& i : result) i = i - second;
		return result;
	}

	matrix operator*(const ValueType& second) {
		matrix result(*this);
		for (auto& i : result.a) i = i * second;
		return result;
	}

	matrix operator/(const ValueType& second) {
		matrix result(*this);
		for (auto& i : result.a) i = i / second;
		return result;
	}

	friend matrix operator*(const matrix& first, const matrix& second) {
		try {
			if (first.tsShape[1] != second.tsShape[0])
				throw "matrix multiplication error: shape error!";
		} catch (const char* str) {
			std::cerr << str << '\n';
			exit(EXIT_FAILURE);
		}
		matrix<ValueType> result(Index({first.tsShape[0], second.tsShape[1]}));
		const size_t &im = first.tsShape[0], &jm = second.tsShape[1], &km = first.tsShape[1];
		for (size_t i = 0; i < im; ++i)
			for (size_t k = 0; k < km; ++k)
				for (size_t j = 0; j < jm; ++j)
					result.a[i * jm + j] += first.a[i * km + k] * second.a[k * jm + j];
		return result;
	}

	void operator=(const matrix& in) {
		a = in.a;
		tsShape = in.tsShape;
		tsShapeSuffix = in.tsShapeSuffix;
	}

	operator tensor<ValueType>() {
		return *((tensor<ValueType>*)this);
	}
};