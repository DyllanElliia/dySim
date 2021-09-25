/*
 * @Author: DyllanElliia
 * @Date: 2021-09-25 16:01:48
 * @LastEditTime: 2021-09-25 16:50:27
 * @LastEditors: DyllanElliia
 * @Description: Syntactic sugar!
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>

// Print the value no line break.
template <typename T, typename... Ts> void qprint_nlb(T v, Ts... vl) {
  std::cout << v << " ";
  if constexpr (sizeof...(vl) > 0) {
    qprint_nlb(vl...);
  }
  return;
}

// Print the value no line break.
template <typename T, typename... Ts> void qprint(T v, Ts... vl) {
  std::cout << v << "\n";
  if constexpr (sizeof...(vl) > 0) {
    qprint(vl...);
  }
  return;
}

inline void qprint() {
  std::cout << std::endl;
  return;
}