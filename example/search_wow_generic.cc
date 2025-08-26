#include "../wow/index.hh"
#include "bench_utils.hh"
#include <omp.h>
#include <atomic>

int main(int argc, char **argv)
{
  std::string base_vec, quer_vec, query_pred, gt_file, index_location, space;
  size_t      k, npass;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--query_vec") == 0) {
      quer_vec = argv[++i];
    } else if (strcmp(argv[i], "--query_pred") == 0) {
      query_pred = argv[++i];
    } else if (strcmp(argv[i], "--base_vec") == 0) {
      base_vec = argv[++i];
    } else if (strcmp(argv[i], "--gt_file") == 0) {
      gt_file = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    } else if (strcmp(argv[i], "--npass") == 0) {
      npass = std::stoul(argv[++i]);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }
  std::cout << "query_vec: " << quer_vec << ", query_pred: " << query_pred << ", gt_file: " << gt_file << ", k: " << k
            << ", index_location: " << index_location << std::endl;

  // load query vectors
  size_t d, nq, nb;
  float *query_vecs = benchmark::fvecs_read(quer_vec, d, nq);
  std::cout << "Loaded query vectors: " << quer_vec << ", d: " << d << ", nq: " << nq << std::endl;
  // load base vectors
  float *base_vecs = benchmark::fvecs_read(base_vec, d, nb);
  std::cout << "Loaded base vectors: " << base_vec << ", d: " << d << ", nb: " << nb << std::endl;
  wowlib::WoWIndex<int, float> index(index_location, space);

  nq = 1000;
  std::vector<wowlib::wow_bitset<wowlib::label_t>> query_bits;
  for (int iq = 0; iq < nq; iq++) {
    query_bits.emplace_back(std::move(benchmark::GenBitmap(npass, nb)));
  }
  std::vector<wowlib::label_t> attvec;
  attvec.resize(nb);
  std::iota(attvec.begin(), attvec.end(), 0);

  std::cout << "query_bits generated" << std::endl;
  auto gt = benchmark::GenGT(nb, nq, d, k, query_bits, base_vecs, query_vecs, attvec, space);
  std::cout << "Ground truth generated" << std::endl;

  std::cout << "searching..." << std::endl;
  std::vector<size_t> efs_list = {1700,
      1400,
      1100,
      1000,
      900,
      800,
      700,
      600,
      500,
      400,
      300,
      250,
      200,
      180,
      160,
      140,
      120,
      100,
      90,
      80,
      70,
      60,
      55,
      50,
      45,
      40,
      35,
      30,
      25,
      20,
      15,
      10};

  for (auto efs : efs_list) {
    std::vector<std::vector<wowlib::label_t>> results(nq);
    float                                     time     = 0;
    float                                     avg_dist = 0;
    float                                     avg_hops = 0;
    index.metric_dist_comps_                           = 0;
    index.metric_hops_                                 = 0;
    for (size_t i = 0; i < nq; ++i) {
      auto start  = std::chrono::high_resolution_clock::now();
      auto result = index.searchKNN(query_vecs + i * d, efs, k, query_bits[i]);
      auto end    = std::chrono::high_resolution_clock::now();
      time += std::chrono::duration<float>(end - start).count();
      for (auto &r : result) {
        results[i].emplace_back(r.second);
      }
    }
    float recall = benchmark::CalculateRecall(gt, results);
    std::cout << efs << "," << recall << "," << nq / time << "," << index.metric_dist_comps_ / nq << ","
              << index.metric_hops_ / nq << std::endl;
  }
  std::cout << "search done" << std::endl;
  delete[] query_vecs;
  delete[] base_vecs;
}