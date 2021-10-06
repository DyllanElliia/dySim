/*
 * @Author: DyllanElliia
 * @Date: 2021-09-15 14:41:40
 * @LastEditTime: 2021-10-06 16:35:29
 * @LastEditors: DyllanElliia
 * @Description: based-modulus
 */

#pragma once

#include "./Index.hpp"

Index addIndex(const Index &i1, const Index &i2, int i2begin = 0) {
  Index result(i1);
  for (size_t i = 0; i < i2.size(); ++i) {
    result[i2begin++] += i2[i];
  }
  return result;
}

template <class T> class Tensor {
protected:
  using ValueType = T;

  std::vector<ValueType> a;
  // Shape of the Tensor user want to create.
  Index tsShape;
  std::vector<ull> tsShapeSuffix;

  // // Real Shape of the Tensor.
  // // e.g. user want to create [4,5,1], ordering to impress the efficiency,
  // the program would create [5,5,1], and Mapping it to [4,5,1].
  // std::vector<int> tsRealShape;

  virtual bool show_(Index &indexS, size_t indexS_i, std::string &outBegin,
                     const std::string &addStr, std::ostringstream &out) {
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
    out << (indexS_i == 0 ? "]" : "],\n");
    return true;
  }

  void updateSuffix() {
    tsShapeSuffix.clear();
    tsShapeSuffix.resize(tsShape.size() + 1, 1);
    for (int i = tsShape.size() - 1; i >= 0; --i) {
      tsShapeSuffix[i] = tsShape[i] * tsShapeSuffix[i + 1];
    }
  }

  static void shapeCheck(const Index &shape1, const Index &shape2) {
    // std::cout << "run sc!" << std::endl;
    // std::cout << shape.size() << std::endl;
    // if (shape.size() != 2)std::cout << "asdf" << std::endl;
    try {
      if (shape1 != shape2)
        throw "\033[1;31mTensor error: Tensors must be equal in shape!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
  }

  template <class tranFun>
  static Tensor computer(const Tensor &first, const Tensor &second) {
    shapeCheck(first.tsShape, second.tsShape);
    Tensor result;
    result.tsShape = first.tsShape;
    result.updateSuffix();
    result.a.reserve(first.a.size());
    std::transform(first.a.begin(), first.a.begin() + first.a.size(),
                   second.a.begin(), std::back_inserter(result.a), tranFun());
    return result;
  }

  void runCut(const Index &from, Tensor &result, const int &ibegin, Index &ci,
              size_t i) {
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
    show_(indexS, 0, outBegin, "|   ", out);
    return out;
  }

public:
  Tensor(const Index &shape, ValueType defaultValue) {
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
  Tensor(const Index &shape, std::function<std::vector<ValueType>()> creatFun) {
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
  Tensor(const Index &shape,
         std::function<std::vector<ValueType>(const Index &shape)> creatFun) {
    tsShape = shape;
    ull sizetsR = 1;
    for (auto i : tsShape)
      sizetsR *= i;
    // for (auto i : tsShape) std::cout << i << " ";
    // printf("\n");
    a = creatFun(shape);
    updateSuffix();
    // std::cout << a.size() << std::endl;
    // for (auto i : a) std::cout << i << " ";
    // std::cout << std::endl;
  }
  Tensor(Tensor<ValueType> &&ts) {
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Tensor(Tensor<ValueType> &ts) {
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Tensor(const ValueType &v) {}
  Tensor() {}
  ~Tensor() {}

  virtual ValueType &operator[](const Index &index_) {
    try {
      if (index_.size() != tsShape.size())
        throw "\033[1;31mTensor error: (Index)Index is not equal to Tensor "
              "shape!\033[0m";
      ull indexR = 0;
      int max_ = index_.size() - 1;
      for (int i = 0; i < max_; ++i) {
        indexR += index_[i] * tsShapeSuffix[i + 1];
      }
      indexR += index_[max_];
      return a[indexR];
    } catch (const char *str) {
      // std::cout << index_.size() << " " << tsShape.size() << std::endl;
      std::cerr << str << '\n';
      return a[0];
    }
  }

  virtual ValueType &operator[](const int &index_) {
    try {
      if (index_ >= tsShapeSuffix[0])
        throw "\033[1;31mTensor error: (int)Index is larger than Tensor "
              "shape\033[0m";

      return a[index_];
    } catch (const char *str) {
      // std::cout << index_.size() << " " << tsShape.size() << std::endl;
      std::cerr << str << '\n';
      return a[0];
    }
  }

  virtual Tensor operator+(const Tensor &ts) {
    return computer<std::plus<ValueType>>(*this, ts);
  }

  virtual Tensor operator-(const Tensor &ts) {
    return computer<std::minus<ValueType>>(*this, ts);
  }

  virtual Tensor operator*(const Tensor &ts) {
    return computer<std::multiplies<ValueType>>(*this, ts);
  }

  virtual Tensor operator/(const Tensor &ts) {
    return computer<std::divides<ValueType>>(*this, ts);
  }

  virtual Tensor operator=(const Tensor &in) {
    // a = in.a;
    // tsShape = in.tsShape;
    // tsShapeSuffix = in.tsShapeSuffix;

    auto &a_ = in.a;
    a.resize(a_.size());
    for (int i = 0, j = 0; i < a_.size(); ++i, ++j)
      a[i] = a_[j];
    // std::cout << a[0] << std::endl;

    tsShape = in.tsShape;
    tsShapeSuffix = in.tsShapeSuffix;
    return *this;
  }

  // friend Tensor operator+(const Tensor &first, const Tensor &second) {
  //   return computer<std::plus<ValueType>>(first, second);
  // }

  friend Tensor operator+(const ValueType &first, Tensor &second) {
    Tensor result(second);
    for (auto &i : result.a)
      i = first + i;
    return result;
  }

  friend Tensor operator+(Tensor &first, const ValueType &second) {
    Tensor result(first);
    for (auto &i : result.a)
      i = i + second;
    return result;
  }

  // friend Tensor operator-(const Tensor &first, const Tensor &second) {
  //   return computer<std::minus<ValueType>>(first, second);
  // }

  friend Tensor operator-(const ValueType &first, Tensor &second) {
    Tensor result(second);
    for (auto &i : result.a)
      i = first - i;
    return result;
  }

  friend Tensor operator-(Tensor &first, const ValueType &second) {
    Tensor result(first);
    for (auto &i : result.a)
      i = i - second;
    return result;
  }

  // void operator=(const Tensor &in) {
  //   a = in.a;
  //   tsShape = in.tsShape;
  //   tsShapeSuffix = in.tsShapeSuffix;
  // }

  Tensor operator*(const ValueType &second) {
    Tensor result(*this);
    for (auto &i : result.a)
      i = i * second;
    return result;
  }
  friend Tensor operator*(const ValueType &first, Tensor &second) {
    Tensor result(second);
    for (auto &i : result.a)
      i = i * first;
    return result;
  }

  Tensor operator/(const ValueType &second) {
    Tensor result(*this);
    for (auto &i : result.a)
      i = i / second;
    return result;
  }

  friend std::ostream &operator<<(std::ostream &output, Tensor &ts) {
    output << ts.getShowOut().str();
    return output;
  }

  class iterator {
  private:
    ValueType *p;

  public:
    iterator(ValueType *p = nullptr) : p(p) {}

    ValueType &operator*() { return *p; }

    // std::vector<ValueType> *operator->() const;

    iterator &operator++() {
      ++p;
      return *this;
    }

    iterator operator++(int) { return iterator(p++); }

    bool operator==(const iterator &arg) const { return p == arg.p; }

    bool operator!=(const iterator &arg) const { return p != arg.p; }
  };

  virtual iterator begin() { return iterator(a.data()); }

  virtual iterator end() { return iterator(a.data() + a.size()); }

  virtual bool show(const std::string &colTabStr = "|   ") {
    /* 		std::cout << "show!" << std::endl;
    std::cout << tsShape[0] << " " << tsShape[1] << std::endl;
    std::cout << (*this)[gi(0, 0)] << std::endl;
    for (auto i : a) std::cout << i << " ";
    std::cout << std::endl; */
    std::ostringstream out;
    Index indexS(tsShape.size(), 0);
    std::string outBegin = "";
    bool result = show_(indexS, 0, outBegin, colTabStr, out);
    // std::cout << "show!beg" << std::endl;
    std::cout << out.str() << std::endl;
    return result;
  }

  // cutting your Tensor!
  // This function is the bottom function of Slice and Fiber.
  virtual Tensor cut(const Index &from, const Index &to) {
    Tensor<ValueType> result;
    int ibegin = 0, iend = from.size() - 1;
    // std::cout << "here" << std::endl;
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

  // return Tensor's shape
  virtual Index shape() { return tsShape; }

  virtual bool reShape(const Index &i) {
    tsShape = i;
    updateSuffix();
    a.resize(tsShapeSuffix[0]);
    return true;
  }
};
