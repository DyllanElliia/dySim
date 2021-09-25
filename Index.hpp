#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "./tools/sugar.hpp"

using Index = std::vector<int>;
using ll = long long;
using ull = unsigned long long;

template <class tranFun>
static Index icomputer(const Index &first, const Index &second) {
  Index result;
  result.reserve(first.size());
  std::transform(first.begin(), first.end(), second.begin(),
                 std::back_inserter(result), tranFun());
  return result;
}

void IndexSizeCheck(const Index &first, const Index &second) {
  try {
    if (first.size() != second.size())
      throw "Index1's shape is not equal to Index2's shape!";
  } catch (char *str) {
    std::cerr << str << '\n';
    exit(EXIT_FAILURE);
  }
}

Index operator+(const Index &i1, const Index &i2) {
  IndexSizeCheck(i1, i2);
  return icomputer<std::plus<int>>(i1, i2);
}

Index operator-(const Index &i1, const Index &i2) {
  IndexSizeCheck(i1, i2);
  return icomputer<std::minus<int>>(i1, i2);
}

Index operator*(const Index &i1, const Index &i2) {
  IndexSizeCheck(i1, i2);
  return icomputer<std::multiplies<int>>(i1, i2);
}

Index operator/(const Index &i1, const Index &i2) {
  IndexSizeCheck(i1, i2);
  return icomputer<std::divides<int>>(i1, i2);
}

// getIndex == gi
// It is a template which can help you packing the long parameters to an Vector.
template <typename... Ints> inline Index gi(Ints... args) {
  int i[] = {(args)...};
  Index a(std::begin(i), std::end(i));
  return a;
}

std::vector<ull> tsShapeSuffix;
Index tsShapeSuffix_equal;

void updateSuffix(const Index &shape) {
  if (shape == tsShapeSuffix_equal)
    return;
  tsShapeSuffix_equal.clear();
  tsShapeSuffix_equal = shape;
  tsShapeSuffix.clear();
  tsShapeSuffix.resize(shape.size() + 1, 1);
  for (int i = shape.size() - 1; i >= 0; --i) {
    tsShapeSuffix[i] = shape[i] * tsShapeSuffix[i + 1];
  }
}

// Index to Tensor Index
// This function can be used to transform Index to array Index
// e.g. i2t1([0,1,2],[3,3,3]) == 5
ull i2ti(const Index &index, const Index &shape) {
  updateSuffix(shape);
  ull indexR = 0;
  int max_ = index.size() - 1;
  for (int i = 0; i < max_; ++i) {
    indexR += index[i] * tsShapeSuffix[i + 1];
  }
  indexR += index[max_];
  return indexR;
}

// printIndex == pi
// This function can transform the index to a string.
std::string pi(const Index index) {
  std::ostringstream out;
  out << "( ";
  for (auto i : index)
    out << i << " ";
  out << ")";
  return out.str();
}