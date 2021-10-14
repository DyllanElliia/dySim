/*
 * @Author: DyllanElliia
 * @Date: 2021-09-25 16:01:48
 * @LastEditTime: 2021-10-13 17:58:38
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
#include <vector>

// Print the values no line break.
template <typename T, typename... Ts> void qprint_nlb(T v, Ts... vl) {
  std::cout << v << " ";
  if constexpr (sizeof...(vl) > 0) {
    qprint_nlb(vl...);
    return;
  }
  std::cout << std::endl;
}

// Print the values with line break.
template <typename T, typename... Ts> void qprint(T v, Ts... vl) {
  std::cout << v << "\n";
  if constexpr (sizeof...(vl) > 0) {
    qprint(vl...);
  }
}

inline void qprint() { std::cout << std::endl; }

#include <ctime>
namespace dym {
class TimeLog {
private:
  double timep;
  std::vector<std::pair<double, double>> timeLogs;
  bool flag;

public:
  void record() {
    flag = false;
    std::cout << "Run time: " << (clock() - timep) / 1000 << "s" << std::endl;
  }

  void reStart() { timep = clock(); }

  void saveLog() {
    timeLogs.push_back(
        std::make_pair(timeLogs.size(), (clock() - timep) / 1000));
  }

  void saveLog(double tag) {
    timeLogs.push_back(std::make_pair(tag, (clock() - timep) / 1000));
  }

  TimeLog() {
    timep = clock();
    flag = true;
  }

  ~TimeLog() {
    if (flag) {
      record();
    }
  }
};
} // namespace dym