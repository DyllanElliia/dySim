#pragma once
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Index = std::vector<int>;
using ll = long long;
using ull = unsigned long long;

template <typename T>
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
	virtual bool show_(Index& indexS, size_t indexS_i, std::string& outBegin, const std::string& addStr) {
		if (indexS_i == tsShape.size()) {
			printf("%d ", (*this)[indexS]);
			return true;
		}
		std::cout << outBegin;
		outBegin += addStr;
		std::cout << "[";
		// run
		if (indexS_i + 1 != tsShape.size())
			std::cout << "\n";
		for (int i = 0; i < tsShape[indexS_i]; ++i) {
			indexS[indexS_i] = i;
			show_(indexS, indexS_i + 1, outBegin, addStr);
		}

		outBegin = outBegin.substr(0, outBegin.size() - addStr.size());
		if (indexS_i + 1 != tsShape.size())
			std::cout << outBegin;
		printf("],\n");
	}
	void updateSuffix() {
		tsShapeSuffix.clear();
		tsShapeSuffix.resize(tsShape.size() + 1, 1);
		for (int i = tsShape.size() - 1; i >= 0; --i) {
			tsShapeSuffix[i] = tsShape[i] * tsShapeSuffix[i + 1];
		}
	}

public:
	tensor(const Index& shape, const ValueType defaultValue) {
		tsShape = shape;
		ull sizetsR = 1;
		for (auto i : tsShape)
			sizetsR *= i;
		for (auto i : tsShape) std::cout << i << " ";
		printf("\n");
		a.resize(sizetsR, defaultValue);
		updateSuffix();
		std::cout << a.size() << std::endl;
	}
	tensor(tensor& ts) {
		tsShape = ts.tsShape;
		a.assign(ts.a.begin(), ts.a.end());
		updateSuffix();
	}
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
			// for (auto i : index_) std::cout << i;
			// std::cout << "(" << indexR << ")[";
			return a[indexR];
		} catch (char* str) {
			std::cerr << str << '\n';
			return a[0];
		}
	}
	virtual bool show(const std::string& colTabStr = "|   ") {
		for (auto i : a) {
			printf("%d ", i);
		}
		std::cout << " end" << std::endl;
		Index indexS(tsShape.size(), 0);
		std::string outBegin = "";
		return show_(indexS, 0, outBegin, colTabStr);
	}
};