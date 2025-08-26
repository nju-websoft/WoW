#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <random>
#include <numeric>
#include "../wow/utils.hh"
#include "../wow/visit_list.hh"

namespace benchmark {
auto fvecs_read(const std::string &filename, size_t &d_out, size_t &n_out) -> float *
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }
  int d;
  in.read(reinterpret_cast<char *>(&d), 4);
  d_out = d;
  // calculate file size
  in.seekg(0, std::ios::beg);
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n    = file_size / (4 + d * sizeof(float));
  n_out       = n;
  float *data = new float[d * n];
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&d), 4);
    in.read(reinterpret_cast<char *>(data + i * d), d * sizeof(float));
  }
  in.close();
  return data;
}

auto LoadRange(const std::string &location) -> std::vector<wowlib::wow_range<int>>
{
  std::vector<wowlib::wow_range<int>> query_filters;
  std::ifstream                       ifs(location, std::ios::binary);
  if (!ifs.is_open()) {
    std::cout << "Fail to open: " << location << std::endl;
    std::abort();
  }
  // check meta size
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  auto n_query = file_size / (2 * sizeof(int));
  query_filters.resize(n_query);
  for (size_t i = 0; i < n_query; ++i) {
    int l, u;
    ifs.read(reinterpret_cast<char *>(&l), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&u), sizeof(int));
    query_filters[i] = wowlib::wow_range<int>{l, u};
  }
  ifs.close();
  // LOG first 10 query filters
  std::string qf_str;
  for (size_t i = 0; i < std::min<size_t>(10, n_query); ++i) {
    qf_str += "[" + std::to_string(query_filters[i].l_) + "," + std::to_string(query_filters[i].u_) + "]";
  }
  std::cout << "First 10 query filters: " << qf_str << std::endl;
  return query_filters;
}

template <typename att_t, typename filter_t>
auto GenGT(size_t nb, size_t nq, size_t d, size_t k, const std::vector<filter_t> &filter, const float *basevec,
    const float *queryvec, std::vector<att_t> attvec, const std::string &space)
    -> std::vector<std::vector<wowlib::label_t>>
{
  wowlib::SpaceInterface<float> *space_ptr;
  if (space == "l2") {
    space_ptr = new wowlib::L2Space(d);
  } else if (space == "ip") {
    space_ptr = new wowlib::InnerProductSpace(d);
  } else {
    throw std::runtime_error("unsupported space type " + space + ", supported: l2, ip");
  }
  auto                                      fstdistfunc     = space_ptr->get_dist_func();
  auto                                      dist_func_param = space_ptr->get_dist_func_param();
  std::vector<std::vector<wowlib::label_t>> gt(nq);
  size_t                                    iq;
  std::cout << "Generating ground truth..." << std::endl;
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic) \
    shared(gt, queryvec, basevec, attvec, fstdistfunc, dist_func_param, k)
  for (iq = 0; iq < nq; ++iq) {
    std::vector<wowlib::dist_id_pair> gt_cand;
    for (int ib = 0; ib < nb; ++ib) {
      if (filter[iq].Test(attvec[ib])) {
        auto dist = fstdistfunc(queryvec + iq * d, basevec + ib * d, dist_func_param);
        PUSH_HEAP(gt_cand, dist, ib);
        if (gt_cand.size() > k) {
          POP_HEAP(gt_cand);
        }
      }
    }

    gt[iq].resize(gt_cand.size());
    for (size_t i = 0; i < gt_cand.size(); ++i) {
      gt[iq][i] = gt_cand[i].id_;
    }
  }
  delete space_ptr;
  return gt;
}

auto LoadBitmap(const std::string &bitmap_file, size_t n) -> std::vector<wowlib::wow_bitset<wowlib::label_t>>
{
  std::ifstream in(bitmap_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + bitmap_file);
  }
  std::vector<wowlib::wow_bitset<wowlib::label_t>> all_bitmap;
  while (!in.eof()) {
    int k;
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    if (in.eof()) {
      break;
    }
    wowlib::wow_bitset<wowlib::label_t> bitmap(n);
    for (int i = 0; i < k; ++i) {
      unsigned int ib;
      in.read(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
      if (ib >= n) {
        throw std::runtime_error("bitmap index out of range: " + std::to_string(ib) + ", n: " + std::to_string(n));
      }
      bitmap.Set(ib);
    }
    all_bitmap.emplace_back(std::move(bitmap));
  }
  std::cout << "Loaded bitmap: " << all_bitmap.size() << std::endl;
  return all_bitmap;
}

auto GenBitmap(size_t npass, size_t nb) -> wowlib::wow_bitset<wowlib::label_t>
{
  // npass is the number of 1 and others are 0 in 0--nb-1
  wowlib::wow_bitset<wowlib::label_t> bitmap(nb);
  bitmap.Clear();
  std::vector<wowlib::label_t>     idx(nb);
  std::iota(idx.begin(), idx.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(idx.begin(), idx.end(), g);
  for (size_t i = 0; i < npass; ++i) {
    bitmap.Set(idx[i]);
  }
  return std::move(bitmap);
}

auto LoadGroundTruth(const std::string &gt_file) -> std::vector<std::vector<wowlib::label_t>>
{
  std::ifstream in(gt_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + gt_file);
  }
  std::vector<std::vector<wowlib::label_t>> all_gt;
  while (!in.eof()) {
    int k;
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    if (in.eof()) {
      break;
    }
    std::vector<wowlib::label_t> gt(k);
    for (int i = 0; i < k; ++i) {
      unsigned int ib;
      in.read(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
      gt[i] = ib;
    }
    all_gt.emplace_back(gt);
  }
  // example
  std::cout << "Loaded ground truth: " << all_gt.size() << std::endl;
  std::cout << "Example: first query has " << all_gt[0].size() << std::endl;
  for (auto ib : all_gt[0]) {
    std::cout << ib << ",";
  }
  std::cout << std::endl;
  return all_gt;
}

template <typename att_t>
auto LoadAttVec(const std::string att_file) -> std::vector<att_t>
{
  std::ifstream in(att_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + att_file);
  }
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n = file_size / sizeof(att_t);
  if (file_size % sizeof(att_t) != 0) {
    throw std::runtime_error("File size is not a multiple of att_t");
  }
  std::vector<att_t> att_vec(n);
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&att_vec[i]), sizeof(att_t));
  }
  in.close();
  std::cout << "Loaded att_vec: " << att_file << ", size: " << n << std::endl;
  return att_vec;
}

auto CalculateRecall(std::vector<std::vector<wowlib::label_t>> &gt, std::vector<std::vector<wowlib::label_t>> &res)
    -> float
{
  size_t n       = std::min(gt.size(), res.size());
  size_t total   = 0;
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    total += gt[i].size();
    for (auto ib : res[i]) {
      if (std::find(gt[i].begin(), gt[i].end(), ib) != gt[i].end()) {
        correct++;
      }
    }
  }
  return static_cast<float>(correct) / total;
}

auto CalculateRecall(std::vector<wowlib::label_t> &gt, std::vector<wowlib::label_t> &res) -> float
{
  size_t total   = gt.size();
  size_t correct = 0;
  for (auto ib : res) {
    if (std::find(gt.begin(), gt.end(), ib) != gt.end()) {
      correct++;
    }
  }
  return static_cast<float>(correct) / total;
}
}  // namespace benchmark