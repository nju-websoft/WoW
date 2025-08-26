#pragma once

#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <unordered_set>
#include <sys/mman.h>

#define PUSH_HEAP(vec, ...)      \
  vec.emplace_back(__VA_ARGS__); \
  std::push_heap(vec.begin(), vec.end())

#define POP_HEAP(vec)                    \
  std::pop_heap(vec.begin(), vec.end()); \
  vec.pop_back();

#define TOP_HEAP(vec) vec.front()

namespace wowlib {
typedef unsigned int       tableint;
typedef int                layer_t;
typedef size_t             label_t;
typedef float              dist_t;
typedef unsigned short int vl_type;

template <typename att_t>
struct wow_range
{
  att_t l_{};
  att_t u_{};

  wow_range(const att_t &l, const att_t &u) : l_(l), u_(u) {}
  wow_range() = default;

  inline __attribute__((always_inline)) auto Test(const att_t &att) const -> bool { return att >= l_ && att <= u_; }
};

template <typename att_t>
struct wow_set
{
  std::unordered_set<att_t>                  set_;
  void                                       Set(att_t i) { set_.insert(i); }
  inline __attribute__((always_inline)) bool Test(att_t i) const { return set_.find(i) != set_.end(); }
};

struct dist_id_pair
{
  dist_t   dist_;
  tableint id_;

  dist_id_pair() = default;

  dist_id_pair(dist_t dist, tableint id) : dist_(dist), id_(id) {}

  bool operator<(const dist_id_pair &rhs) const { return dist_ < rhs.dist_; }

  bool operator>(const dist_id_pair &rhs) const { return dist_ > rhs.dist_; }
};
}  // namespace wowlib