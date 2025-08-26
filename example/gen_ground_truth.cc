#include "../wow/space_dist.hh"
#include "bench_utils.hh"

#include <numeric>

int main(int argc, char **argv)
{
  std::string basevecfile, queryvecfile, gtfile, attfile, queryfilterfile, space;
  size_t      k;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--basevec") == 0) {
      basevecfile = argv[++i];
    } else if (strcmp(argv[i], "--queryvec") == 0) {
      queryvecfile = argv[++i];
    } else if (strcmp(argv[i], "--gt_file") == 0) {
      gtfile = argv[++i];
    } else if (strcmp(argv[i], "--att_file") == 0) {
      attfile = argv[++i];
    } else if (strcmp(argv[i], "--query_rng") == 0) {
      queryfilterfile = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    }
  }
  size_t           d, nb, nq;
  auto             basevec     = benchmark::fvecs_read(basevecfile, d, nb);
  auto             queryvec    = benchmark::fvecs_read(queryvecfile, d, nq);
  auto             queryfilter = benchmark::LoadRange(queryfilterfile);
  std::vector<int> attvec;
  if (attfile == "serial") {
    attvec.resize(nb);
    std::iota(attvec.begin(), attvec.end(), 0);
  } else {
    attvec = benchmark::LoadAttVec<int>(attfile);
  }
  if (attvec.size() != nb) {
    throw std::runtime_error("attvec size is not equal to basevec size");
  }
  if (queryfilter.size() != nq) {
    // throw std::runtime_error("queryfilter size is not equal to queryvec size");
    nq = std::min(nq, queryfilter.size());
  }
  std::ofstream os(gtfile, std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("Cannot open file " + gtfile);
  }
  // generate ground truth whose att is in query range
  auto gt = benchmark::GenGT(nb, nq, d, k, queryfilter, basevec, queryvec, attvec, space);
  for (size_t iq = 0; iq < nq; ++iq) {
    k = gt[iq].size();
    os.write(reinterpret_cast<char *>(&k), sizeof(int));
    for (size_t i = 0; i < k; ++i) {
      unsigned int ib = gt[iq][i];
      os.write(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
    }
  }

  os.close();
  std::cout << "Ground truth generated: " << gtfile << std::endl;
  delete[] basevec;
  delete[] queryvec;
}