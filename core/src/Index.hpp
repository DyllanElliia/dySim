#pragma once

#include "../../tools/sugar.hpp"
#include "./define.hpp"

namespace dym {

template <class ValueType = int> struct Index {
  ValueType *a;
  int rank;
  Index() : a(NULL), rank(0) {}
  Index(ValueType *a_, int _rank) : rank(_rank) {
    a = new ValueType[_rank];
    for (int i = 0; i < _rank; ++i)
      a[i] = a_[i];
  }
  Index(int _rank, ValueType v = 0) : rank(_rank) {
    a = new ValueType[_rank];
    for (int i = 0; i < _rank; ++i)
      a[i] = v;
  }
  ~Index() { delete[] a; }
  Index(const Index &i_) {
    rank = i_.rank;
    a = new ValueType[rank];
    for (int i = 0; i < rank; ++i)
      a[i] = i_.a[i];
  }
  Index(const Index &&i_) {
    rank = i_.rank;
    a = new ValueType[rank];
    for (int i = 0; i < rank; ++i)
      a[i] = i_.a[i];
  }

  _DYM_GENERAL_ ValueType &operator[](const int &Index_) const {
    return a[Index_ % rank];
  }
  _DYM_GENERAL_ Index operator=(const Index &in) {
    if (a)
      delete[] a;
    rank = in.rank;
    a = new ValueType[rank];
    for (int i = 0; i < rank; ++i)
      a[i] = in.a[i];
    return *this;
  }
  _DYM_GENERAL_ bool operator==(const Index &arg) const {
    const auto &a1 = a, &a2 = arg.a;
    if (rank != arg.rank)
      return false;
    for (int i = 0; i < rank; ++i)
      if (a1[i] != a2[i])
        return false;
    return true;
  }

  _DYM_GENERAL_ bool operator!=(const Index &arg) const {
    const auto &a1 = a, &a2 = arg.a;
    if (rank != arg.rank)
      return false;
    for (int i = 0; i < rank; ++i)
      if (a1[i] == a2[i])
        return false;
    return true;
  }

  inline _DYM_GENERAL_ int size() const { return rank; }
  inline _DYM_GENERAL_ void clear() {
    delete[] a;
    rank = 0;
  }
  inline _DYM_GENERAL_ void resize(int rank_, ValueType v = 0) {
    rank = rank_;
    a = new ValueType[rank];
    for (int i = 0; i < rank; ++i)
      a[i] = v;
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
    _DYM_GENERAL_ iterator operator++(ValueType) { return iterator(p++); }
    _DYM_GENERAL_ bool operator==(const iterator &arg) const {
      return p == arg.p;
    }
    _DYM_GENERAL_ bool operator!=(const iterator &arg) const {
      return p != arg.p;
    }
  };
  _DYM_GENERAL_ iterator begin() { return iterator(a); }
  _DYM_GENERAL_ iterator end() { return iterator(a + rank); }
  _DYM_GENERAL_ void push_back(ValueType v) {
    ValueType *p = new ValueType[rank + 1];
    for (int i = 0; i < rank; ++i)
      p[i] = a[i];
    p[rank++] = v;
    if (a)
      delete[] a;
    a = p;
  }
};
using ll = long long;
using ull = unsigned long long;

// getIndex == gi
// It is a template which can help you packing the long parameters to an Vector.
template <typename... Ints> inline _DYM_GENERAL_ Index<int> gi(Ints... args) {
  int i[] = {int(args)...};
  // Index a(std::begin(i), std::end(i));
  return Index(i, sizeof(i) / sizeof(int));
}

// printIndex == pi
// This function can transform the Index to a string.
template <typename t = int> std::string pi(Index<t> index) {
  std::ostringstream out;
  out << "( ";
  for (auto i : index)
    out << i << " ";
  out << ")";
  return out.str();
}

Index<ull> tsShapeSuffix;
Index<int> tsShapeSuffix_equal;

template <typename t = int> void updateSuffix(const Index<t> &shape) {
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
template <typename t> ull i2ti(const Index<t> &index, const Index<t> &shape) {
  updateSuffix(shape);
  ull IndexR = 0;
  int max_ = index.size() - 1;
  for (int i = 0; i < max_; ++i) {
    IndexR += index[i] * tsShapeSuffix[i + 1];
  }
  IndexR += index[max_];
  return IndexR;
}

template <class func, typename t>
static _DYM_GENERAL_ Index<t> icomputer(const Index<t> &first,
                                        const Index<t> &second, func tranFun) {
  auto rank1 = first.size(), rank2 = second.size();
  Index<t> result;
  if (rank1 > rank2) {
    result.a = new int[rank1];
    result.rank = rank1;
  } else {
    result.a = new int[rank2];
    result.rank = rank2;
  }
  // result.reserve(first.size());
  // std::transform(first.begin(), first.end(), second.begin(),
  //                result, tranFun());
  for (int i = 0; i < rank1; ++i)
    result[i] = tranFun(first[i], second[i]);
  return result;
}

template <typename t = int>
_DYM_GENERAL_ void IndexSizeCheck(const Index<t> &first,
                                  const Index<t> &second) {
  if (first.size() != second.size()) {
    const char str[] = "Index1's shape is not equal to Index2's shape!";
    printf("%s\nshape: %d, %d", str, first.size(), second.size());
    exit(EXIT_FAILURE);
  }
}
} // namespace dym

#define dym_Index_operator(op)                                                 \
  template <typename t = int>                                                  \
  inline _DYM_GENERAL_ dym::Index<t> operator op(const dym::Index<t> &i1,      \
                                                 const dym::Index<t> &i2) {    \
    IndexSizeCheck(i1, i2);                                                    \
    return icomputer(i1, i2, [] _DYM_LAMBDA_(const t &a1, const t &a2) -> t {  \
      return a1 op a2;                                                         \
    });                                                                        \
  }

dym_Index_operator(+)

    dym_Index_operator(-)

        dym_Index_operator(*)

            dym_Index_operator(/)

    // dym::Index operator+(const
    // dym::Index &i1, const dym::Index
    // &i2) {
    //   dym::IndexSizeCheck(i1, i2);
    //   return
    //   dym::icomputer<std::plus<int>>(i1,
    //   i2);
    // }

    // dym::Index operator-(const
    // dym::Index &i1, const dym::Index
    // &i2) {
    //   dym::IndexSizeCheck(i1, i2);
    //   return
    //   dym::icomputer<std::minus<int>>(i1,
    //   i2);
    // }

    // dym::Index operator*(const
    // dym::Index &i1, const dym::Index
    // &i2) {
    //   dym::IndexSizeCheck(i1, i2);
    //   return
    //   dym::icomputer<std::multiplies<int>>(i1,
    //   i2);
    // }

    // dym::Index operator/(const
    // dym::Index &i1, const dym::Index
    // &i2) {
    //   dym::IndexSizeCheck(i1, i2);
    //   return
    //   dym::icomputer<std::divides<int>>(i1,
    //   i2);
    // }
