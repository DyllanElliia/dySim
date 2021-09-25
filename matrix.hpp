/*
 * @Author: DyllanElliia
 * @Date: 2021-09-17 14:02:29
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-25 22:37:14
 * @Description:
 */

#pragma once
#include "./Tensor.hpp"

template <class T> class Matrix : public Tensor<T> {
protected:
  using ValueType = T;
  using ValueUse = T &;
  using ValuePtr = T *;

  using Tensor<ValueType>::a;
  using Tensor<ValueType>::tsShape;
  using Tensor<ValueType>::tsShapeSuffix;
  using Tensor<ValueType>::updateSuffix;
  // using Tensor<ValueType>::operator+;
  // template <class tranFun, class comValue>
  // using computer = Tensor<ValueType>::computer<tranFun, comValue>;

  inline void shapeCheck(const Index &shape) {
    // std::cout << "run sc!" << std::endl;
    // std::cout << shape.size() << std::endl;
    // if (shape.size() != 2)std::cout << "asdf" << std::endl;
    try {
      if (shape.size() != 2)
        throw "\033[1;31mMatrix error: Matrix shape must be "
              "2-dimensional!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
  }

  template <class tranFun>
  static Matrix computer(const Matrix &first, const Matrix &second) {
    Matrix result;
    result.tsShape = first.tsShape;
    result.updateSuffix();
    result.a.reserve(first.a.size());
    std::transform(first.a.begin(), first.a.end(), second.a.begin(),
                   std::back_inserter(result.a), tranFun());
    return result;
  }

public:
  Matrix(const Index &shape, ValueType defaultValue = 0)
      : Tensor<ValueType>(shape, defaultValue) {
    shapeCheck(shape);
  }
  Matrix(const Index &shape, std::function<std::vector<ValueType>()> creatFun)
      : Tensor<ValueType>(shape, creatFun) {
    shapeCheck(shape);
  }
  Matrix(const Index &shape,
         std::function<std::vector<ValueType>(const Index &shape)> creatFun)
      : Tensor<ValueType>(shape, creatFun) {
    shapeCheck(shape);
  }
  Matrix(Matrix<ValueType> &&ts) : Tensor<ValueType>() {
    shapeCheck(ts.tsShape);
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Matrix(Matrix<ValueType> &ts) : Tensor<ValueType>() {
    // std::cout << "in" << std::endl;
    shapeCheck(ts.tsShape);
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Matrix(Tensor<ValueType> &&ts) : Tensor<ValueType>() {
    shapeCheck(ts.tsShape);
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Matrix(Tensor<ValueType> &ts) : Tensor<ValueType>() {
    shapeCheck(ts.tsShape);
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Matrix(std::vector<std::vector<ValueType>> &v) : Tensor<ValueType>() {
    tsShape.push_back(v.size());
    int my = 1e7;
    for (auto &l : v)
      my = std::min(my, (int)l.size());
    tsShape.push_back(my);
    for (auto &l : v)
      a.insert(a.end(), l.begin(), l.begin() + my);
    updateSuffix();
  }
  Matrix() : Tensor<ValueType>() {}
  ~Matrix() {}

  friend Matrix operator+(const Matrix &first, const Matrix &second) {
    return computer<std::plus<ValueType>>(first, second);
  }

  /* 	Matrix operator+(const ValueType& second) {
          Matrix result(*this);
          for (auto& i : result.a) i = i + second;
          return result;
  } */

  friend Matrix operator+(const ValueType &first, Matrix &second) {
    Matrix result(second);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = first + a_[i];
    return result;
  }

  friend Matrix operator+(Matrix &first, const ValueType &second) {
    Matrix result(first);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = a_[i] + second;
    return result;
  }

  friend Matrix operator-(const Matrix &first, const Matrix &second) {
    return computer<std::minus<ValueType>>(first, second);
  }

  friend Matrix operator-(const ValueType &first, Matrix &second) {
    Matrix result(second);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = first - a_[i];
    return result;
  }

  friend Matrix operator-(Matrix &first, const ValueType &second) {
    Matrix result(first);
    auto &a_ = result.a;
    // std::cout << first.a[0] << " c" << std::endl;
    // std::cout << first.a.size() << " " << a_.size() << std::endl;
    // std::cout << a_[0] << std::endl;
    // std::cout << typeid(second).name() << std::endl;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = a_[i] - second;
    // std::cout << a_[0] << std::endl;
    return result;
  }

  Matrix operator*(const ValueType &second) {
    Matrix result(*this);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = a_[i] * second;
    return result;
  }

  friend Matrix operator*(const ValueType &first, Matrix &second) {
    Matrix result(second);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = first * a_[i];
    return result;
  }

  Matrix operator/(const ValueType &second) {
    Matrix result(*this);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = a_[i] / second;
    return result;
  }

  friend Matrix operator/(const ValueType &first, Matrix &second) {
    Matrix result(second);
    auto &a_ = result.a;
    for (int i = 0; i < a_.size(); ++i)
      a_[i] = first / a_[i];
    return result;
  }

  friend Matrix operator*(const Matrix &first, const Matrix &second) {
    try {
      if (first.tsShape[1] != second.tsShape[0])
        throw "\033[1;31mMatrix multiplication error: shape error!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
    Matrix<ValueType> result(Index({first.tsShape[0], second.tsShape[1]}));
    const size_t &im = first.tsShape[0], &jm = second.tsShape[1],
                 &km = first.tsShape[1];
    for (size_t i = 0; i < im; ++i)
      for (size_t k = 0; k < km; ++k)
        for (size_t j = 0; j < jm; ++j)
          result.a[i * jm + j] += first.a[i * km + k] * second.a[k * jm + j];
    return result;
  }

  Matrix &operator=(const Matrix &in) {
    // a = in.a;
    // std::cout << "here=" << std::endl;
    // std::cout << in.a[0] << std::endl;
    auto &a_ = in.a;
    a.resize(a_.size());
    for (int i = 0, j = 0; i < a_.size(); ++i, ++j)
      a[i] = a_[j];
    // std::cout << a[0] << std::endl;

    tsShape = in.tsShape;
    tsShapeSuffix = in.tsShapeSuffix;
    return *this;
  }

  operator Tensor<ValueType>() { return *((Tensor<ValueType> *)this); }

  bool reShape(const Index &i) {
    // std::cout << a.size() << " h" << std::endl;
    int ar = i[0] * i[1];
    // std::cout << i[0] << " " << i[1] << " " << ar << std::endl;
    a.resize(ar);
    tsShape = i;
    // std::cout << a.size() << " h" << std::endl;
    updateSuffix();
    return true;
  }

  int text() { return a.size(); }
};