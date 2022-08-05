/*
 * @Author: DyllanElliia
 * @Date: 2021-10-06 17:00:50
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-12 15:59:50
 * @Description:
 */

#pragma once

// quat
#include "math/quaternion.hpp"
// real
#include "math/realALG.hpp"
// vec
#include "math/vecALG.hpp"
#include "math/vector.hpp"
// mat
#include "math/matALG.hpp"
#include "math/matrix.hpp"
// dual
#include "math/dual_alg.hpp"
#include "math/dual_num.hpp"
// tensor
#include "math/tensor.hpp"

namespace dym {
typedef Matrix<Real, 3, 3> Matrix3;
typedef Matrix<Real, 4, 4> Matrix4;
typedef Matrix<lReal, 3, 3> Matrix3l;
typedef Matrix<lReal, 4, 4> Matrix4l;
typedef Matrix<Reali, 3, 3> Matrix3i;
typedef Matrix<Reali, 4, 4> Matrix4i;
typedef Vector<Real, 2> Vector2;
typedef Vector<Real, 3> Vector3;
typedef Vector<Real, 4> Vector4;
typedef Vector<lReal, 2> Vector2l;
typedef Vector<lReal, 3> Vector3l;
typedef Vector<lReal, 4> Vector4l;
typedef Vector<Reali, 2> Vector2i;
typedef Vector<Reali, 3> Vector3i;
typedef Vector<Reali, 4> Vector4i;
template <class InputType> Tensor<InputType> abs(Tensor<InputType> in) {
  in.for_each_i([](InputType &e) { e = dym::max(e, (InputType)0); });
  return in;
}

template <typename ValueType>
Tensor<ValueType> cross(const Tensor<ValueType> &A,
                        const Tensor<ValueType> &B) {
  auto As = A.shape(), Bs = B.shape();
  try {
    if (As.rank != 2 || Bs.rank != 2)
      throw "\033[1;31mDyMath cross() error: Tensor must is matrix!\033[0m]";
  } catch (const char *str) {
    std::cerr << str << '\n';
    return Tensor<ValueType>(0, dym::gi(0));
  }

  Tensor<ValueType> result(0, dym::gi(As[0], Bs[1]));
  const int &flen = As[1] == Bs[0] ? As[1] : 0, &bs1 = Bs[1];
  result.for_each_i([&](ValueType &e, int i, int j) {
    for (int ii = 0; ii < flen; ++ii)
      e += A[i * flen + ii] * B[ii * bs1 + j];
  });
  return result;
}
}; // namespace dym