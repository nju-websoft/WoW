#include "../wow/index.hh"
#include "bench_utils.hh"
#include <omp.h>
#include <atomic>

int main(int argc, char **argv)
{
  size_t      m, efc, dim, maxN;
  std::string basevec, baseatt, space, index_location;
  int         t;
  size_t      o = 4, wp = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--m") == 0) {
      m = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--efc") == 0) {
      efc = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--basevec") == 0) {
      basevec = argv[++i];
    } else if (strcmp(argv[i], "--baseatt") == 0) {
      baseatt = argv[++i];
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    } else if (strcmp(argv[i], "--threads") == 0) {
      t = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    } else if (strcmp(argv[i], "--o") == 0) {
      o = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--wp") == 0) {
      wp = std::stoul(argv[++i]);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }
  std::cout << "m: " << m << ", efc: " << efc << ", basevec: " << basevec << ", o: " << o << ", wp: " << wp
            << ", space: " << space << std::endl;
  // load base vectors
  float                       *basevecs = benchmark::fvecs_read(basevec, dim, maxN);
  wowlib::WoWIndex<int, float> index(maxN, dim, m, efc, space, o, wp, wp == 0);
  std::vector<int>             ids(maxN);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), std::mt19937{std::random_device{}()});
  std::vector<int> att_vec;
  if (baseatt == "serial") {
    att_vec.resize(maxN);
    std::iota(att_vec.begin(), att_vec.end(), 0);
  } else {
    att_vec = benchmark::LoadAttVec<int>(baseatt);
  }
  auto start = std::chrono::high_resolution_clock::now();
  //   std::atomic<size_t> counter;
#pragma omp parallel for num_threads(t) schedule(dynamic) shared(index)
  for (size_t i = 0; i < maxN; ++i) {
    auto cur_id = ids[i];
    index.insert(cur_id, basevecs + cur_id * dim, att_vec[cur_id]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Index built in " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
  // save index
  index.save(index_location);
  std::cout << "Index saved to: " << index_location << std::endl;
  delete[] basevecs;
}