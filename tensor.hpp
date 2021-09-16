#pragma once
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using Index = std::vector<int>;
using ll = long long;
using ull = unsigned long long;

Index addIndex(const Index& i1, const Index& i2, int i2begin = 0) {
	Index result(i1);
	for (int i = 0; i < i2.size(); ++i) {
		result[i2begin++] += i2[i];
	}
	return result;
}

template <class T>
class tensor {
protected:
	using ValueType = T;
	using ValueUse = T&;
	using ValuePtr = T*;

	std::vector<ValueType> a;
	// Shape of the tensor user want to create.
	std::vector<int> tsShape;
	std::vector<ull> tsShapeSuffix;
	// // Real Shape of the tensor.
	// // e.g. user want to create [4,5,1], ordering to impress the efficiency, the program would create [5,5,1], and Mapping it to [4,5,1].
	// std::vector<int> tsRealShape;

private:
	virtual bool show_(Index& indexS, size_t indexS_i, std::string& outBegin, const std::string& addStr, std::ostringstream& out) {
		if (indexS_i == tsShape.size()) {
			out << (*this)[indexS] << " ";
			return true;
		}
		out << outBegin;
		outBegin += addStr;
		out << "[ ";
		// run
		if (indexS_i + 1 != tsShape.size())
			out << "\n";
		for (int i = 0; i < tsShape[indexS_i]; ++i) {
			indexS[indexS_i] = i;
			show_(indexS, indexS_i + 1, outBegin, addStr, out);
		}

		outBegin = outBegin.substr(0, outBegin.size() - addStr.size());
		if (indexS_i + 1 != tsShape.size())
			out << outBegin;
		out << "],\n";
	}

	void updateSuffix() {
		tsShapeSuffix.clear();
		tsShapeSuffix.resize(tsShape.size() + 1, 1);
		for (int i = tsShape.size() - 1; i >= 0; --i) {
			tsShapeSuffix[i] = tsShape[i] * tsShapeSuffix[i + 1];
		}
	}

	template <class tranFun>
	static tensor computer(const tensor& first, const tensor& second) {
		tensor result;
		result.tsShape = first.tsShape;
		result.updateSuffix();
		result.a.reserve(first.a.size());
		std::transform(first.a.begin(), first.a.end(), second.a.begin(), std::back_inserter(result.a), tranFun());
		return result;
	}

	void runCut(const Index& from, tensor& result, const int& ibegin, Index& ci, size_t i) {
		if (i == ci.size()) {
			result[ci] = (*this)[addIndex(from, ci, ibegin)];
			return;
		}
		for (int num = 0; num < result.tsShape[i]; num++) {
			ci[i] = num;
			runCut(from, result, ibegin, ci, i + 1);
		}
	}

	std::ostringstream getShowOut() {
		std::ostringstream out;
		Index indexS(tsShape.size(), 0);
		std::string outBegin = "";
		bool result = show_(indexS, 0, outBegin, "|   ", out);
		return out;
	}

public:
	tensor(const Index& shape, const ValueType defaultValue) {
		tsShape = shape;
		ull sizetsR = 1;
		for (auto i : tsShape)
			sizetsR *= i;
		// for (auto i : tsShape) std::cout << i << " ";
		// printf("\n");
		a.resize(sizetsR, defaultValue);
		updateSuffix();
		// std::cout << a.size() << std::endl;
	}
	tensor(const Index& shape, std::function<std::vector<ValueType>()> creatFun) {
		tsShape = shape;
		ull sizetsR = 1;
		for (auto i : tsShape)
			sizetsR *= i;
		// for (auto i : tsShape) std::cout << i << " ";
		// printf("\n");
		a = creatFun();
		updateSuffix();
		// std::cout << a.size() << std::endl;
	}
	tensor(tensor<ValueType>&& ts) {
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
	tensor(tensor<ValueType>& ts) {
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
	tensor() {}
	~tensor() {}

	ValueType& operator[](const Index& index_) {
		try {
			if (index_.size() != tsShape.size())
				throw "Index is not equal to tensor shape";
			ull indexR = 0;
			int max_ = index_.size() - 1;
			for (int i = 0; i < max_; ++i) {
				indexR += index_[i] * tsShapeSuffix[i + 1];
			}
			indexR += index_[max_];
			return a[indexR];
		} catch (char* str) {
			std::cerr << str << '\n';
			return a[0];
		}
	}

	friend tensor operator+(const tensor& first, const tensor& second) {
		return computer<std::plus<ValueType>>(first, second);
	}

	friend tensor operator-(const tensor& first, const tensor& second) {
		return computer<std::minus<ValueType>>(first, second);
	}

	void operator=(const tensor& in) {
		a = in.a;
		tsShape = in.tsShape;
		tsShapeSuffix = in.tsShapeSuffix;
	}

	friend std::ostream& operator<<(std::ostream& output, tensor& ts) {
		output << ts.getShowOut().str();
		return output;
	}

	virtual bool show(const std::string& colTabStr = "|   ") {
		std::ostringstream out;
		Index indexS(tsShape.size(), 0);
		std::string outBegin = "";
		bool result = show_(indexS, 0, outBegin, colTabStr, out);
		std::cout << out.str();
		return result;
	}

	// cutting your tensor!
	// This function is the bottom function of Slice and Fiber.
	virtual tensor cut(const Index& from, const Index& to) {
		tensor<ValueType> result;
		int ibegin = 0, iend = from.size() - 1;
		std::cout << "here" << std::endl;
		while (ibegin < iend) {
			if (from[ibegin] + 1 == to[ibegin]) {
				ibegin++;
				continue;
			}
			if (from[ibegin] + 1 > to[ibegin])
				return result;
			break;
		}
		while (ibegin < iend) {
			if (from[iend] == to[iend]) {
				iend--;
				continue;
			}
			if (from[iend] > to[iend])
				return result;
			break;
		}
		++iend;
		int tsShapeSize = iend - ibegin;
		if (tsShapeSize <= 0)
			return result;
		result.tsShape.resize(tsShapeSize);
		ull len = 1;
		for (int i = ibegin, ri = 0; i < iend; ++i, ++ri) {
			result.tsShape[ri] = to[i] - from[i];
			len *= result.tsShape[ri];
		}
		result.updateSuffix();
		result.a.resize(len);
		Index ci(tsShapeSize, 0);
		runCut(from, result, ibegin, ci, 0);
		return result;
	}

	// return tensor's shape
	Index shape() {
		return tsShape;
	}
};
