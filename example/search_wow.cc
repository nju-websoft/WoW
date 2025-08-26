#include "../wow/index.hh"
#include "bench_utils.hh"
#include <omp.h>
#include <atomic>

int main(int argc, char **argv)
{
  std::string quer_vec, query_rng, gt_file, index_location, space;
  size_t      k;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--query_vec") == 0) {
      quer_vec = argv[++i];
    } else if (strcmp(argv[i], "--query_rng") == 0) {
      query_rng = argv[++i];
    } else if (strcmp(argv[i], "--gt_file") == 0) {
      gt_file = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    }else{
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }
  std::cout << "query_vec: " << quer_vec << ", query_rng: " << query_rng << ", gt_file: " << gt_file << ", k: " << k
            << ", index_location: " << index_location << std::endl;

  // load query vectors
  size_t d, nq;
  float *query_vecs = benchmark::fvecs_read(quer_vec, d, nq);
  std::cout << "Loaded query vectors: " << quer_vec << ", d: " << d << ", nq: " << nq << std::endl;
  // load query filters
  std::vector<wowlib::wow_range<int>> query_filters = benchmark::LoadRange(query_rng);
  std::cout << "Loaded query filters: " << query_rng << std::endl;
  // load ground truth
  std::vector<std::vector<wowlib::label_t>> gt = benchmark::LoadGroundTruth(gt_file);
  std::cout << "Loaded ground truth: " << gt_file << std::endl;
  // load index
  wowlib::WoWIndex<int, float> index(index_location, space);

  nq = 1000;

  std::cout << "searching..." << std::endl;
  std::vector<size_t> efs_list = {1700,1400,1100,1000,900,800,700,600,500,400,300,250,200,180,
    160,140,120,100,90,80,70,60,55,50,45,40,35,30,25,20,15,10};
  for(auto efs: efs_list){
    std::vector<std::vector<wowlib::label_t>> results(nq);
    float time = 0;
    float avg_dist = 0;
    float avg_hops = 0;
    index.metric_dist_comps_ = 0;
    index.metric_hops_ = 0;
    for (size_t i = 0; i < nq; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      auto result = index.searchKNN(query_vecs + i * d, efs, k, query_filters[i]);
      auto end   = std::chrono::high_resolution_clock::now();
      time += std::chrono::duration<float>(end - start).count();
      for(auto &r : result) {
        results[i].emplace_back(r.second);
      }
    }
    float recall = benchmark::CalculateRecall(gt,results);
    std::cout << efs<<","<<recall<<","<<nq/time<<","<<index.metric_dist_comps_/nq<<","<<index.metric_hops_/nq<<std::endl;
  }
  std::cout << "search done" << std::endl;
  delete[] query_vecs;
}