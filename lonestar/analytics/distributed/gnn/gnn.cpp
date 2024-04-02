#include "DistBench/Output.h"
#include "DistBench/Start.h"
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "galois/DReducible.h"
#include "galois/DTerminationDetector.h"
#include "galois/runtime/Tracer.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

enum { CPU };
int personality = CPU;

constexpr static const char* const REGION_NAME = "GNN";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<float> tolerance("tolerance",
                                 cll::desc("tolerance for residual"),
                                 cll::init(0.000001));
static cll::opt<unsigned int>
    maxIterations("maxIterations", cll::desc("Maximum iterations: Default 10"),
                  cll::init(10));

enum Exec { Sync, Async };
static cll::opt<Exec> execution(
    "exec", cll::desc("Distributed Execution Model (default value Async):"),
    cll::values(clEnumVal(Sync, "Bulk-synchronous Parallel (BSP)"),
                clEnumVal(Async, "Bulk-asynchronous Parallel (BASP)")),
    cll::init(Sync));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

#define GNNL 6
typedef float VEC[GNNL];
struct WeightDist {
  float w[GNNL * GNNL];
};

static const float w_global[] = {
    0.5900383917267924,    0.18093935695476637, -0.22508656567864627,
    0.25795760566598713,   0.7171742875157778,  0.6901366016144754,
    -0.026496391637854444, -0.6230933119488107, -0.49048605137331114,
    -0.8030693756798981,   0.7054218182489014,  -0.7430840919671189,
    -0.9425971502847392,   0.6953654406900365,  -0.6532535885249664,
    -0.2476721861806157};

struct NodeData {
  VEC value1;
  std::atomic<uint32_t> nout;
  WeightDist w;
};

galois::DynamicBitSet bitset_nout;

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

std::unique_ptr<galois::graphs::GluonSubstrate<Graph>> syncSubstrate;

#include "gnn_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

float rand_float() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

/* (Re)initialize all fields to 0 except for residual which needs to be 0.15
 * everywhere */
struct ResetGraph {
  Graph* graph;
  ResetGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
        galois::iterate(allNodes.begin(), allNodes.end()), ResetGraph{&_graph},
        galois::no_stats(),
        galois::loopname(
            syncSubstrate->get_run_identifier("ResetGraph").c_str()));
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    // init labels
    for (int i = 0; i < GNNL; i++) {
      sdata.value1[i] = rand_float();
    }
    // init out neighbors
    sdata.nout = 0;
    // init weights
    for (int i = 0; i < GNNL * GNNL; i++) {
      //      sdata.w.w[i] = w_global[i];
      sdata.w.w[i] = rand_float();
    }
  }
};

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    // init graph
    ResetGraph::go(_graph);

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    // doing a local do all because we are looping over edges
    galois::do_all(
        galois::iterate(nodesWithEdges), InitializeGraph{&_graph},
        galois::steal(), galois::no_stats(),
        galois::loopname(
            syncSubstrate->get_run_identifier("InitializeGraph").c_str()));

    syncSubstrate
        ->sync<writeDestination, readAny, Reduce_add_nout, Bitset_nout>(
            "InitializeGraph");
  }

  // Calculate "outgoing" edges for destination nodes (note we are using
  // the tranpose graph for pull algorithms)
  void operator()(GNode src) const {
    for (auto nbr : graph->edges(src)) {
      GNode dst   = graph->getEdgeDst(nbr);
      auto& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, (uint32_t)1);
    }
  }
};

// TODO: GPU code operator does not match CPU's operator (cpu accumulates sum
// and adds all at once, GPU adds each pulled value individually/atomically)

template <bool async>
struct GNN {
  Graph* graph;

  using DGTerminatorDetector =
      typename std::conditional<async, galois::DGTerminator<unsigned int>,
                                galois::DGAccumulator<unsigned int>>::type;

  GNN(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    unsigned _num_iterations   = 0;
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    DGTerminatorDetector dga;

    // unsigned int reduced = 0;

    do {
      syncSubstrate->set_num_round(_num_iterations);
      // reset residual on mirrors
      galois::do_all(galois::iterate(nodesWithEdges), GNN{&_graph},
                     galois::steal(), galois::no_stats(),
                     galois::loopname(
                         syncSubstrate->get_run_identifier("Forward").c_str()));

      galois::runtime::reportStat_Tsum(
          REGION_NAME, "NumWorkItems_" + (syncSubstrate->get_run_identifier()),
          (unsigned long)_graph.sizeEdges());

      ++_num_iterations;
      std::cout << "GNN Iteration " << _num_iterations << "\n";
    } while (async || (_num_iterations < maxIterations));

    galois::runtime::reportStat_Tmax(
        REGION_NAME,
        "NumIterations_" + std::to_string(syncSubstrate->get_run_num()),
        (unsigned long)_num_iterations);
  }

  void spmv(WeightDist& w, VEC& src, VEC& dst) const {
    for (int i = 0; i < GNNL; i++) {
      dst[i]     = 0;
      float* w_i = w.w + i * GNNL;
      for (int j = 0; j < GNNL; j++) {
        dst[i] += w_i[j] * src[j];
      }
    }
  }

  void add(VEC& dst, VEC& src) const {
    for (int i = 0; i < GNNL; i++) {
      dst[i] += src[i];
    }
  }

  void divide(VEC& dst, uint32_t nout) const {
    for (int i = 0; i < GNNL; i++) {
      dst[i] /= nout;
    }
  }

  // Pull deltas from neighbor nodes, then add to self-residual
  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    VEC tmp     = {0, 0, 0, 0};
    for (auto nbr : graph->edges(src)) {
      GNode dst   = graph->getEdgeDst(nbr);
      auto& ddata = graph->getData(dst);
      add(tmp, ddata.value1);
    }
    add(tmp, sdata.value1);
    divide(tmp, sdata.nout);
    spmv(sdata.w, tmp, sdata.value1);
    //    std::cout << "done: " << src << "\n";
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "GNN - Compiler Generated "
                                          "Distributed Heterogeneous";
constexpr static const char* const desc = "GNN Pull version on "
                                          "Distributed Galois.";
constexpr static const char* const url  = nullptr;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::runtime::reportParam(REGION_NAME, "Max Iterations", maxIterations);
    std::ostringstream ss;
    ss << tolerance;
    galois::runtime::reportParam(REGION_NAME, "Tolerance", ss.str());
  }

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

  std::unique_ptr<Graph> hg;

  std::tie(hg, syncSubstrate) =
      distGraphInitialization<NodeData, void, false>();

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  InitializeGraph::go(*hg);
  galois::runtime::getHostBarrier().wait();

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] GNN::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();

    GNN<false>::go(*hg);

    StatTimer_main.stop();

    if ((run + 1) != numRuns) {
      bitset_nout.reset();

      syncSubstrate->set_num_run(run + 1);
      InitializeGraph::go(*hg);
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  if (output) {
    //    std::vector<float> results = makeResults(hg);
    //    auto globalIDs = hg->getMasterGlobalIDs();
    //    assert(results.size() == globalIDs.size());

    //    writeOutput(outputLocation, "gnn", results.data(), results.size(),
    //                globalIDs.data());
  }

  return 0;
}
