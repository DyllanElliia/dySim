/*
 * @Author: DyllanElliia
 * @Date: 2022-03-09 15:16:14
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-09 16:28:34
 * @Description:
 */
#include <dyMath.hpp>

int main(int argc, char const *argv[]) {
  dym::Quaternion q = dym::getQuaternion<Real>(dym::Pi / 2.f, {0.f, 1.f, 0.f});
  q.show();
  (q * 2 + 3).show();
  dym::Quaternion q2 = dym::getQuaternion<Real>(dym::Pi / 2.f, {1.f, 0.f, 0.f});
  q2.show();
  auto matR = q.to_matrix();
  matR = matR * q2.to_matrix();
  matR.show();
  dym::Vector3 v({1.f, 0.f, 1.f});
  (matR * v).show();
  ((q * q2).to_matrix() * v).show();
  return 0;
}
