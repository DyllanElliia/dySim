/*
 * @Author: DyllanElliia
 * @Date: 2021-09-15 14:41:40
 * @LastEditTime: 2022-03-01 16:54:49
 * @LastEditors: DyllanElliia
 * @Description: based-modulus
 */

#pragma once

#include "./Launch.hpp"
#include "./vector.hpp"
namespace dym {
template <typename t>
Index<t> addIndex(const Index<t> &i1, const Index<t> &i2, int i2begin = 0) {
  Index<t> result(i1);
  for (size_t i = 0; i < i2.size(); ++i) {
    result[i2begin++] += i2[i];
  }
  return result;
}

template <typename T>
constexpr _DYM_FORCE_INLINE_ bool is_calculated() {
  return std::is_same_v<T, Real> || std::is_same_v<T, float> ||
         std::is_same_v<T, double> || std::is_same_v<T, int> ||
         std::is_same_v<T, short> || std::is_same_v<T, long long>;
}

template <class T>
class Tensor {
 protected:
  using ValueType = T;
  using shapeType = int;

  std::vector<ValueType> a;
  // Shape of the Tensor user want to create.
  Index<shapeType> tsShape;
  Index<ull> tsShapeSuffix;

  // // Real Shape of the Tensor.
  // // e.g. user want to create [4,5,1], ordering to impress the efficiency,
  // the program would create [5,5,1], and Mapping it to [4,5,1].
  // std::vector<int> tsRealShape;

  virtual bool show_(Index<shapeType> &indexS, size_t indexS_i,
                     std::string &outBegin, const std::string &addStr,
                     std::ostringstream &out) {
    if (indexS_i == tsShape.size()) {
      out << (*this)[indexS] << " ";
      return true;
    }
    out << outBegin;
    outBegin += addStr;
    out << "[ ";
    // run
    if (indexS_i + 1 != tsShape.size()) out << "\n";
    for (int i = 0; i < tsShape[indexS_i]; ++i) {
      indexS[indexS_i] = i;
      show_(indexS, indexS_i + 1, outBegin, addStr, out);
    }

    outBegin = outBegin.substr(0, outBegin.size() - addStr.size());
    if (indexS_i + 1 != tsShape.size()) out << outBegin;
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

  static void shapeCheck(const Index<shapeType> &shape1,
                         const Index<shapeType> &shape2) {
    try {
      if (shape1 != shape2)
        throw "\033[1;31mTensor error: Tensors must be equal in shape!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
  }

  template <class func>
  static Tensor computer(const Tensor &first, const Tensor &second,
                         func tranFun) {
    // shapeCheck(first.tsShape, second.tsShape);
    if (first.tsShape == second.tsShape) {
      Tensor result;
      result.tsShape = first.tsShape;
      result.updateSuffix();
      result.a.resize(first.a.size(), 0);
      result.for_each_i([&first, &second, &tranFun](ValueType &e, int i) {
        e = tranFun(first.a[i], second.a[i]);
      });
      // std::transform(first.a.begin(), first.a.begin() + first.a.size(),
      //                second.a.begin(), std::back_inserter(result.a),
      //                tranFun());
      return result;
    }
    Tensor t1, t2;
    t1 = first, t2 = second;
    if (t1.tsShape.size() < t2.tsShape.size()) std::swap(t1, t2);
    auto &t1s = t1.tsShape, &t2s = t2.tsShape;
    Tensor result;
    result.tsShape = t1s;
    result.updateSuffix();
    result.a.resize(t1.a.size(), 0);
    try {
      for (int i = 0, t1l = t1s.size() - t2s.size(); i < t2s.size(); ++i, ++t1l)
        if (t1s[t1l] != t2s[i]) {
          if (t2s[i] == 1) {
            auto &t1sl = t1s[t1l];
            Tensor asdf;
            asdf.tsShape = t2s;
            asdf.tsShape[i] = t1sl;
            asdf.updateSuffix();
            asdf.a.resize(t2.a.size() * t1sl);
            auto &aa = asdf.a, &t2a = t2.a;
            auto t2as = t2.a.size();
            for (unsigned int jj = 0; jj < t2as; ++jj) {
              for (unsigned int ii = 0; ii < t1sl; ++ii) {
                aa[ii + jj * t1sl] = t2a[jj];
              }
            }
            t2 = asdf;
            // qprint("test:", t2);
          } else
            throw "\033[1;31mTensor error: Tensors must be equal in "
                  "shape!\033[0m";
        }
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
    int t2len = t2.tsShapeSuffix[0];
    auto &t1end = t1.tsShapeSuffix[0];
    result.for_each_i([&t2len, &t1, &t2, &tranFun](ValueType &e, int i) {
      e = tranFun(t1.a[i], t2.a[i % t2len]);
    });
    // for (int i = 0, ii = t2len; i < t1end; i += t2len, ii += t2len) {
    //   std::transform(t1.a.begin() + i, t1.a.begin() + ii, t2.a.begin(),
    //                  std::back_inserter(result.a), tranFun());
    // }
    return result;
  }

  void runCut(const Index<shapeType> &from, Tensor &result, const int &ibegin,
              Index<shapeType> &ci, size_t i) {
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
    std::string outBegin = "Tensor: ";
    show_(indexS, 0, outBegin, "| ", out);
    return out;
  }

 public:
  Tensor(ValueType defaultValue, const Index<shapeType> &shape) {
    tsShape = shape;
    ull sizetsR = 1;
    for (auto i : tsShape) sizetsR *= i;
    a.resize(sizetsR, defaultValue);
    updateSuffix();
  }
  Tensor(const Index<shapeType> &shape,
         std::function<std::vector<ValueType>()> creatFun) {
    tsShape = shape;
    ull sizetsR = 1;
    for (auto i : tsShape) sizetsR *= i;
    a = creatFun();
    updateSuffix();
  }
  Tensor(const Index<shapeType> &shape,
         std::function<std::vector<ValueType>(const Index<shapeType> &shape)>
             creatFun) {
    tsShape = shape;
    ull sizetsR = 1;
    for (auto i : tsShape) sizetsR *= i;
    a = creatFun(shape);
    updateSuffix();
  }
  Tensor(const Tensor<ValueType> &&ts) {
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  Tensor(const Tensor<ValueType> &ts) {
    tsShape = ts.tsShape;
    a.assign(ts.a.begin(), ts.a.end());
    updateSuffix();
  }
  // Tensor(ValueType v) {
  //   tsShape = gi(1);
  //   updateSuffix();
  //   a.push_back(v);
  // }
  Tensor(const std::vector<std::vector<ValueType>> &v) {
    if (v.size() != 1) tsShape.push_back(v.size());
    int my = 1e7;
    for (auto &l : v) my = std::min(my, (int)l.size());
    tsShape.push_back(my);
    for (auto &l : v) a.insert(a.end(), l.begin(), l.begin() + my);
    updateSuffix();
  }
  Tensor(const std::vector<ValueType> &v, bool t) {
    int len = v.size();
    if (t)
      tsShape = gi(len, 1);
    else
      tsShape = gi(len);
    qprint(pi(tsShape));
    a.assign(v.begin(), v.begin() + len);
    updateSuffix();
  }
  Tensor() {}
  ~Tensor() {}

  virtual Tensor t() {
    auto newShape = tsShape;
    int im, jm = tsShape[0];
    try {
      switch (newShape.size()) {
        case 1:
          im = 1;
          newShape.push_back(1);
          break;
        case 2:
          im = tsShape[1];
          std::swap(newShape[0], newShape[1]);
          break;
        default:
          qprint(newShape.size());
          throw "\033[1;31mTensor error: function t can only be applied to "
              "transposed tensor with dimensions up to 2!\nIf you want to "
              "transpose this tensor, please use function "
              "transpose(times)!\033[0m";
      }
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
    Tensor result(0, newShape);
    for (int i = 0; i < im; ++i)
      for (int j = 0; j < jm; ++j) {
        result[i * jm + j] = a[i + j * im];
      }
    return result;
  }

  virtual Tensor transpose(unsigned int i1 = 0, unsigned int i2 = 1) {
    auto newShape = tsShape;
    unsigned int im, jm;
    if (newShape.size() <= 2) return t();
    try {
      if (i1 > i2) std::swap(i1, i2);
      if (i1 + 1 != i2)
        throw "\033[1;31mTensor error: function transpose only works when "
              "both input parameters must be continuous!\033[0m";
      if (i2 >= tsShape.size())
        throw "\033[1;31mTensor error: function transpose only works when "
              "both input parameters are less than shape size!\033[0m";
      im = newShape[i1], jm = newShape[i2];
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
    std::swap(newShape[i1], newShape[i2]);
    Tensor result(0, newShape);
    // qprint("test::::", result);
    auto &h2ts = result.tsShapeSuffix[i1],
         &rtails = result.tsShapeSuffix[i2 + 1],
         &rmids = result.tsShapeSuffix[i2];
    auto &otails = tsShapeSuffix[i2 + 1], &omids = tsShapeSuffix[i2];
    auto heads = result.tsShapeSuffix[0] / h2ts;
    auto &ra = result.a, &oa = a;
    for (unsigned int h = 0; h < heads; ++h)
      for (unsigned int i = 0; i < im; ++i)
        for (unsigned int j = 0; j < jm; ++j) {
          ull hh = h * h2ts;
          ull rbe = hh + i * rtails + j * rmids;
          ull obe = hh + i * omids + j * otails;
          for (unsigned int k = 0; k < rtails; ++k) ra[rbe + k] = oa[obe + k];
        }
    return result;
  }

  virtual ValueType &operator[](const Index<shapeType> &index_) {
    try {
      if (index_.size() != tsShape.size())
        throw "\033[1;31mTensor error: (Index)Index is not equal to Tensor "
              "shape!\033[0m";
      ull indexR = 0;
      int max_ = index_.size() - 1;
      for (int i = 0; i < max_; ++i) {
        indexR += (index_[i]) % tsShape[i] * tsShapeSuffix[i + 1];
      }
      indexR += index_[max_] % tsShape[max_];
      return a[indexR];
    } catch (const char *str) {
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
      std::cerr << str << '\n';
      return a[index_ % a.size()];
    }
  }

  virtual ValueType operator[](const Index<shapeType> &index_) const {
    try {
      if (index_.size() != tsShape.size())
        throw "\033[1;31mTensor error: (Index)Index is not equal to Tensor "
              "shape!\033[0m";
      ull indexR = 0;
      int max_ = index_.size() - 1;
      for (int i = 0; i < max_; ++i) {
        indexR += (index_[i]) % tsShape[i] * tsShapeSuffix[i + 1];
      }
      indexR += index_[max_] % tsShape[max_];
      return a[indexR];
    } catch (const char *str) {
      std::cerr << str << '\n';
      return a[0];
    }
  }

  virtual ValueType operator[](const int &index_) const {
    try {
      if (index_ >= tsShapeSuffix[0])
        throw "\033[1;31mTensor error: (int)Index is larger than Tensor "
              "shape\033[0m";

      return a[index_];
    } catch (const char *str) {
      std::cerr << str << '\n';
      return a[index_ % a.size()];
    }
  }

  virtual Tensor operator=(const Tensor &in) {
    auto &a_ = in.a;
    a.resize(a_.size());
    tsShape = in.tsShape;
    tsShapeSuffix = in.tsShapeSuffix;
    (*this).for_each_i([&a_](ValueType &e, int i) { e = a_[i]; });
    return *this;
  }
  virtual Tensor operator=(const ValueType &in) {
    (*this).for_each_i([&in](ValueType &e) { e = in; });
    return *this;
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

  virtual Tensor &for_each_i(
      std::function<void(ValueType &)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      for (unsigned int i = ib; i < ie; ++i) {
        auto &tsi = ts[i];
        func(tsi);
      }
    };
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each_i(
      std::function<void(ValueType &, int i)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      for (unsigned int i = ib; i < ie; ++i) {
        func(ts[i], i);
      }
    };
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each_i(
      std::function<void(ValueType &, int i, int j)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      for (unsigned int i = ib; i < ie; ++i) {
        func(ts[i], i / ts.tsShapeSuffix[1], (i % ts.tsShapeSuffix[1]));
      }
    };
    try {
      if (tsShape.size() != 2) {
        if (tsShape.size() > 2) {
          for (unsigned int i = 2; i < tsShape.size(); ++i)
            if (tsShape[i] != 1)
              throw "\033[1;31mTensor error: Only 2-dimensions Tensors can use "
                    "this function!\033[0m";
        } else
          throw "\033[1;31mTensor error: Only 2-dimensions Tensors can use "
                "this function!\033[0m";
      }
    } catch (const char *str) {
      std::cerr << str << '\n'
                << "\033[1;31mYour Tensor's dimensions is " +
                       std::to_string(tsShape.size()) + ".\033[0m\n";
      exit(EXIT_FAILURE);
    }
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each_i(
      std::function<void(ValueType &, int i, int j, int k)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      for (unsigned int i = ib; i < ie; ++i) {
        func(ts[i], i / ts.tsShapeSuffix[1],
             (i % ts.tsShapeSuffix[1]) / ts.tsShapeSuffix[2],
             (i % ts.tsShapeSuffix[2]));
      }
    };
    try {
      if (tsShape.size() != 3) {
        if (tsShape.size() > 3) {
          for (unsigned int i = 3; i < tsShape.size(); ++i)
            if (tsShape[i] != 1)
              throw "\033[1;31mTensor error: Only 3-dimensions Tensors can use "
                    "this function!\033[0m";
        } else
          throw "\033[1;31mTensor error: Only 3-dimensions Tensors can use "
                "this function!\033[0m";
      }
    } catch (const char *str) {
      std::cerr << str << '\n'
                << "\033[1;31mYour Tensor's dimensions is " +
                       std::to_string(tsShape.size()) + ".\033[0m\n";
      exit(EXIT_FAILURE);
    }
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each(
      std::function<void(ValueType *, int i)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      const auto &step = ts.tsShapeSuffix[1];
      for (unsigned int i = ib; i < ie; i += step) {
        func(&(ts[i]), i / ts.tsShapeSuffix[1]);
      }
    };
    try {
      if (tsShape.size() < 2)
        throw "\033[1;31mTensor error: Only >= 2-dimensions Tensors can use "
              "this function!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n'
                << "\033[1;31mYour Tensor's dimensions is " +
                       std::to_string(tsShape.size()) + ".\033[0m\n";
      exit(EXIT_FAILURE);
    }
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each(
      std::function<void(ValueType *, int i, int j)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      for (unsigned int i = ib; i < ie; ++i) {
        func(&(ts[i]), i / ts.tsShapeSuffix[1],
             (i % ts.tsShapeSuffix[1]) / ts.tsShapeSuffix[2]);
      }
    };
    try {
      if (tsShape.size() < 2)
        throw "\033[1;31mTensor error: Only >= 2-dimensions Tensors can use "
              "this function!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n'
                << "\033[1;31mYour Tensor's dimensions is " +
                       std::to_string(tsShape.size()) + ".\033[0m\n";
      exit(EXIT_FAILURE);
    }
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each(
      std::function<void(ValueType *, int i, int j, int k)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      for (unsigned int i = ib; i < ie; ++i) {
        func(&(ts[i]), i / ts.tsShapeSuffix[1],
             (i % ts.tsShapeSuffix[1]) / ts.tsShapeSuffix[2],
             (i % ts.tsShapeSuffix[2]) / ts.tsShapeSuffix[3]);
      }
    };
    try {
      if (tsShape.size() < 3)
        throw "\033[1;31mTensor error: Only >= 3-dimensions Tensors can use "
              "this function!\033[0m";
    } catch (const char *str) {
      std::cerr << str << '\n'
                << "\033[1;31mYour Tensor's dimensions is " +
                       std::to_string(tsShape.size()) + ".\033[0m\n";
      exit(EXIT_FAILURE);
    }
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual Tensor &for_each_i(
      std::function<void(ValueType &, Index<shapeType> &i)> func,
      const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
    auto &ts = *this;
    auto forI = [&ts, &func](const unsigned int ib, const unsigned int ie) {
      auto tsS = ts.tsShape.size();
      auto tsS2 = tsS - 1;
      auto tsSuff = ts.tsShapeSuffix;
      for (unsigned int i = ib; i < ie; ++i) {
        Index<shapeType> in(tsS, i);
        for (unsigned int j = 1; j < tsS; ++j) in[j] %= tsSuff[j];
        for (unsigned int j = 0; j < tsS2; ++j) in[j] /= tsSuff[j + 1];
        func(ts[i], in);
      }
    };
    Launch(forI, 0, tsShapeSuffix[0], use_thread_type);
    return *this;
  }

  virtual bool show(const std::string &colTabStr = "| ") {
    std::ostringstream out;
    Index<shapeType> indexS(tsShape.size(), 0);
    std::string outBegin = "Tensor: ";
    bool result = show_(indexS, 0, outBegin, colTabStr, out);
    std::cout << out.str() << std::endl;
    return result;
  }

  // cutting your Tensor!
  // This function is the bottom function of Slice and Fiber.
  virtual Tensor cut(const Index<shapeType> &from, const Index<shapeType> &to) {
    try {
      if (from.size() != to.size())
        throw "\033[1;31mTensor error: The function cut accepts two Indexes of "
              "equal length\033[0m";
      for (int i = 0; i < from.size(); ++i)
        if (from[i] >= to[i])
          throw "\033[1;31mTensor error: The function cut accepts two Indexes "
                "which Index_from's elements have to be less than Index_to's "
                "elements\033[0m";
    } catch (const char *str) {
      // std::cout << index_.size() << " " << tsShape.size() << std::endl;
      std::cerr << str << '\n';
      return Tensor();
    }
    Tensor<ValueType> result;
    int ibegin = 0, iend = from.size() - 1;
    // std::cout << "here" << std::endl;
    while (ibegin < iend) {
      if (from[ibegin] + 1 == to[ibegin]) {
        ibegin++;
        continue;
      }
      if (from[ibegin] + 1 > to[ibegin]) return result;
      break;
    }
    while (ibegin < iend) {
      if (from[iend] == to[iend]) {
        iend--;
        continue;
      }
      if (from[iend] > to[iend]) return result;
      break;
    }
    // qprint("here");
    ++iend;
    int tsShapeSize = iend - ibegin;
    if (tsShapeSize <= 0) return result;
    result.tsShape.resize(tsShapeSize);
    ull len = 1;
    for (int i = ibegin, ri = 0; i < iend; ++i, ++ri) {
      result.tsShape[ri] = to[i] - from[i];
      len *= result.tsShape[ri];
    }
    result.updateSuffix();
    result.a.resize(len);
    Index<shapeType> ci(tsShapeSize, 0);
    runCut(from, result, ibegin, ci, 0);
    return result;
  }

  // return Tensor's shape
  virtual Index<shapeType> shape() const { return tsShape; }

  virtual bool reShape(const Index<shapeType> &i) {
    tsShape = i;
    updateSuffix();
    a.resize(tsShapeSuffix[0]);
    return true;
  }

#define _dym_tensor_operator_binary_(op)                                    \
  friend Tensor operator op(const ValueType &first, const Tensor &second) { \
    Tensor result(second);                                                  \
    result.for_each_i([&first](ValueType &e) { e = first op e; });          \
    return result;                                                          \
  }                                                                         \
  friend Tensor operator op(const Tensor &first, const ValueType &second) { \
    Tensor result(first);                                                   \
    result.for_each_i([&second](ValueType &e) { e = e op second; });        \
    return result;                                                          \
  }

#define _dym_tentensor_operator_binary_(op)                                 \
  virtual Tensor operator op(const Tensor &ts) {                            \
    return computer(*this, ts, [](const ValueType &a, const ValueType &b) { \
      return a op b;                                                        \
    });                                                                     \
  }

#define _dym_tensor_operator_unary_(op)                             \
  friend void operator op(Tensor &first, const ValueType &second) { \
    first.for_each_i([&second](ValueType &e) { e op second; });     \
  }

  _dym_tentensor_operator_binary_(+);
  _dym_tentensor_operator_binary_(-);
  _dym_tentensor_operator_binary_(*);
  _dym_tentensor_operator_binary_(/);

  // Calculation
  _dym_tensor_operator_binary_(+);
  _dym_tensor_operator_binary_(-);
  _dym_tensor_operator_binary_(*);
  _dym_tensor_operator_binary_(/);
  // _dym_tensor_operator_binary_(%);
  _dym_tensor_operator_unary_(+=);
  _dym_tensor_operator_unary_(-=);
  _dym_tensor_operator_unary_(*=);
  _dym_tensor_operator_unary_(/=);
  // _dym_tensor_operator_unary_(%=);
  // // Logic
  // _dym_tensor_operator_binary_(<<);
  // _dym_tensor_operator_binary_(>>);
  // _dym_tensor_operator_binary_(&);
  // _dym_tensor_operator_binary_(|);
  // _dym_tensor_operator_binary_(^);
  // _dym_tensor_operator_unary_(<<=);
  // _dym_tensor_operator_unary_(>>=);
  // _dym_tensor_operator_unary_(&=);
  // _dym_tensor_operator_unary_(|=);
  // _dym_tensor_operator_unary_(^=);
};
}  // namespace dym