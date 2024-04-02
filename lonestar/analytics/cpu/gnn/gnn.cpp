#include "Lonestar/BoilerPlate.h"
#include "gnn.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/gstl.h"

const char* desc = "Graph Neural Network";

static cll::opt<bool>
    transposedGraph("transposedGraph",
                    cll::desc("Specify that the input graph is transposed"),
                    cll::init(false));

constexpr static const unsigned CHUNK_SIZE = 32;

struct LNode {
  GNNTy value;
  uint32_t nout;
  void init() {
    for (int i = 0; i < GNNL; i++) {
      value[i] = randFloat();
    }
  }
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

void initNodeData(Graph& g) {
  galois::do_all(
      galois::iterate(g),
      [&](GNode n) {
        LNode& data = g.getData(n);
        data.init();
      },
      galois::no_stats(), galois::loopname("init"));
}

void computeOutDeg(Graph& graph) {
  galois::StatTimer outDegreeTimer("computeOutDegFunc");
  outDegreeTimer.start();

  galois::LargeArray<std::atomic<size_t>> vec;
  vec.allocateInterleaved(graph.size());

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) { vec.constructAt(src, 0ul); }, galois::no_stats(),
      galois::loopname("InitDegVec"));

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        for (auto nbr : graph.edges(src)) {
          GNode dst = graph.getEdgeDst(nbr);
          vec[dst].fetch_add(1ul);
        };
      },
      galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
      galois::loopname("computeOutDeg"));

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        auto& srcData = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        srcData.nout  = vec[src];
      },
      galois::no_stats(), galois::loopname("CopyDeg"));

  outDegreeTimer.stop();
}

void Spmv(Weight& w, GNNTy& src, GNNTy& dst) {
  for (int i = 0; i < GNNL; i++) {
    dst[i] = 0;
    for (int j = 0; j < GNNL; j++) {
      dst[i] += w.w[i][j] * src[j];
    }
  }
}

void Add(GNNTy& dst, GNNTy& src) {
  for (int i = 0; i < GNNL; i++) {
    dst[i] += src[i];
  }
}

void Divide(GNNTy& src, uint32_t nout) {
  for (int i = 0; i < GNNL; i++) {
    src[i] /= nout;
  }
}

void ComputeGnn(Graph& graph, Weight& weight) {
  unsigned int iter = 0;
  galois::GAccumulator<unsigned int> accum;

  while (true) {
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          LNode& srcData = graph.getData(src);
          GNNTy newVal;
          for (auto nbr : graph.edges(src)) {
            GNode dst      = graph.getEdgeDst(nbr);
            LNode& dstData = graph.getData(dst);
            Add(newVal, dstData.value);
          }
          Divide(newVal, srcData.nout);
          Spmv(weight, newVal, srcData.value);
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
        galois::loopname("ForwardPass"));
    iter++;
    if (iter >= maxIterations) {
      break;
    }
  }
}

int main(int argc, char** argv) {
  // init random seed for random number generation
  srand(0);
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  Graph transposeGraph;
  std::cout << "WARNING: pull style algorithms work on the transpose of the "
               "actual graph\n"
            << "WARNING: this program assumes that " << inputFile
            << " contains transposed representation\n\n"
            << "Reading graph: " << inputFile << "\n";
  galois::graphs::readGraph(transposeGraph, inputFile);
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";
  galois::preAlloc(2 * numThreads + (3 * transposeGraph.size() *
                                     sizeof(typename Graph::node_data_type)) /
                                        galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  // gnn logic
  Weight weight;
  weight.init();
  initNodeData(transposeGraph);

  computeOutDeg(transposeGraph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  ComputeGnn(transposeGraph, weight);
  execTime.stop();

  galois::reportPageAlloc("MeminfoPost");
  totalTime.stop();
  return 0;
}