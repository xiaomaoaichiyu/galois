#ifndef LONESTAR_GNN_H
#define LONESTAR_GNN_H

#include <iostream>
#include <cstdlib>
#include <ctime>

#define DEBUG 0
#define GNNL 4
#define GNN_K 1

constexpr static const unsigned MAX_ITER = 10;

static const char* name = "Graph Neural Network";
static const char* url  = nullptr;

//! All Graph Neural Network algorithm variants use the same constants for ease
//! of

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<unsigned int> maxIterations(
    "maxIterations",
    cll::desc("Maximum iterations, applies round-based versions only"),
    cll::init(MAX_ITER));

typedef float GNNTy[GNNL];

float randFloat() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

struct Weight {
  ~Weight() {
    if (w) {
      for (int i = 0; i < GNNL; i++) {
        delete[] w[i];
      }
      delete[] w;
    }
  }
  void init() {
    w = new float*[GNNL];
    for (int i = 0; i < GNNL; i++) {
      w[i] = new float[GNNL];
      for (int j = 0; j < GNNL; j++) {
        w[i][j] = randFloat();
      }
    }
  }
public:
  float** w;
};

#if DEBUG
template <typename Graph>
void printPageRank(Graph& graph) {
  std::cout << "Id\tPageRank\n";
  int counter = 0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ii++) {
    auto& sd = graph.getData(*ii);
    std::cout << counter << " " << sd.value << "\n";
    counter++;
  }
}
#endif

#endif // LONESTAR_GNN_H