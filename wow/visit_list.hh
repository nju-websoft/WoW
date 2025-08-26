#pragma once
#include <mutex>
#include "utils.hh"

namespace wowlib {
template <typename id_t = tableint>
class VisitedBaseClass
{
public:
  virtual void Clear()            = 0;
  virtual void Set(id_t i)        = 0;
  virtual bool Test(id_t i) const = 0;
  virtual void Reset(id_t i)      = 0;
  virtual ~VisitedBaseClass()     = default;
};

// memory optimized Bitset, aligned to 64 bits and use aligned alloc for cache line alignment
template <typename id_t = tableint>
class wow_bitset : public VisitedBaseClass<id_t>
{
public:
  wow_bitset() = delete;
  wow_bitset(size_t n) : n_(n)
  {
    size_t n_bytes = (n + 7) / 8;
    size_t aligned_bytes = (n_bytes + 63) & ~63; // Round up to multiple of 64
    data_          = static_cast<uint64_t *>(aligned_alloc(64, aligned_bytes));
    if (data_ == nullptr) {
      throw std::runtime_error("fail to alloc for bitset");
    }
  }

  wow_bitset(const wow_bitset &) = delete;

  wow_bitset(wow_bitset &&other)
  {
    n_          = other.n_;
    data_       = other.data_;
    other.data_ = nullptr;
  }

  ~wow_bitset() override
  {
    if (data_)
      free(data_);
  }

  inline __attribute__((always_inline)) void Set(id_t i) override { data_[i / 64] |= 1ULL << (i % 64); }

  inline __attribute__((always_inline)) bool Test(id_t i) const override { return data_[i / 64] & (1ULL << (i % 64)); }

  inline __attribute__((always_inline)) void Reset(id_t i) override { data_[i / 64] &= ~(1ULL << (i % 64)); }

  inline __attribute__((always_inline)) auto GetData(id_t i) -> uint64_t * { return &data_[i / 64]; }

  inline __attribute__((always_inline)) void Clear() override
  {
    size_t n_bytes = (n_ + 7) / 8;
    memset(data_, 0, n_bytes);
  }

public:
  size_t    n_{};
  uint64_t *data_{nullptr};
};

template <typename id_t = tableint>
class VisitedList : public VisitedBaseClass<id_t>
{
public:
  VisitedList(int numelements1)
  {
    curV_        = -1;
    numelements_ = numelements1;
    size_t total_bytes = numelements_ * sizeof(vl_type);
    size_t aligned_bytes = (total_bytes + 63) & ~63; // Round up to multiple of 64
    mass_        = static_cast<vl_type *>(aligned_alloc(64, aligned_bytes));
    if (mass_ == nullptr) {
      throw std::runtime_error("Failed to allocate memory for VisitedList");
    }
  }

  inline void Clear() override
  {
    curV_++;
    if (curV_ == 0) {
      memset(mass_, 0, sizeof(vl_type) * numelements_);
      curV_++;
    }
  }

  inline __attribute__((always_inline)) void Set(id_t i) override { mass_[i] = curV_; }

  inline __attribute__((always_inline)) bool Test(id_t i) const override { return (mass_[i] == curV_); }

  inline __attribute__((always_inline)) void Reset(id_t i) override { mass_[i] = -1; }

  inline __attribute__((always_inline)) auto GetData(id_t i) -> vl_type * { return &mass_[i]; }

  ~VisitedList() override { free(mass_); }

public:
  vl_type      curV_;
  vl_type     *mass_;
  unsigned int numelements_;
};

template <typename VisitedType = wow_bitset<tableint>>
class VisitedPool
{
public:
  /**
   * @brief Construct a new Visited Bit Set Pool object
   *
   * @param n number of elements for each bitset to store
   * @param pool_size number of bitset to store
   */
  VisitedPool() = default;

  ~VisitedPool()
  {
    for (auto bs : pool_) {
      delete bs;
    }
  }

  void Init(size_t n) { n_ = n; }

  inline __attribute__((always_inline)) auto Get() -> VisitedType *
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pool_.empty()) {
      return new VisitedType(n_);
    }
    auto bs = pool_.back();
    pool_.pop_back();
    return bs;
  }

  inline void Return(VisitedType *bs)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    pool_.push_back(bs);
  }

private:
  size_t                     n_{};
  std::vector<VisitedType *> pool_;
  std::mutex                 mtx_;
};
}  // namespace wowlib