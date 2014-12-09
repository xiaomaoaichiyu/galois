/** Galois Network Layer for Generalized Software Routing -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Network.h"
#include "NetworkIO_mpi.h"

#include <boost/intrusive/list.hpp>

#include <thread>

using namespace Galois::Runtime;

namespace {

template<typename BaseIO>
class NetworkInterfaceRouted : public NetworkInterface {
  static const int COMM_MIN = 1400; // bytes (sligtly smaller than an ethernet packet)
  static const int COMM_DELAY = 100; //microseconds

  struct rmessage : public boost::intrusive::list_base_hook<>{
    std::chrono::high_resolution_clock::time_point time;
    uint32_t dest;
    SendBuffer buf;
    unsigned size() const { return sizeof(dest) + buf.size(); }
    
    rmessage(uint32_t d, SendBuffer&& b)
      : time(std::chrono::high_resolution_clock::now()), dest(d), buf(std::move(b)) {}
    rmessage(uint32_t d, RecvBuffer&& b)
      : time(std::chrono::high_resolution_clock::now()), dest(d), buf(std::move(b)) {}
    rmessage(uint32_t d, const char* data, uint32_t len)
      :time(std::chrono::high_resolution_clock::now()), dest(d), buf(data, len) {}
  };

  struct dataBuffer {
    Galois::Runtime::LL::SimpleLock lock;
    boost::intrusive::list<rmessage> pending;
    unsigned bytes;
    std::atomic<int> urgent;

    dataBuffer() :bytes(0), urgent(0) {}

    std::vector<char> createBuffer(bool& rurgent) {
      std::vector<char> retval(bytes);
      auto ii = retval.begin();
      while (!pending.empty()) {
        auto& n = pending.front();
        char* src = (char*)&n.dest;
        std::copy_n(src, sizeof(n.dest), ii);
        ii += sizeof(n.dest);
        uint32_t psize = n.buf.size();
        assert(psize);
        src = (char*)&psize;
        std::copy_n(src, sizeof(psize), ii);
        ii += sizeof(psize);
        std::copy_n((char*)n.buf.linearData(), n.buf.size(), ii);
        ii += n.buf.size();
        pending.pop_front();
        delete &n;
      }
      // if (ii != retval.end())
      //   std::cerr << "E: " << std::distance(ii, retval.end()) << "\n";
      assert(ii == retval.end());
      rurgent = urgent;
      urgent = 0;
      bytes = 0;
      assert(pending.empty());
      return retval;
    }

    static boost::intrusive::list<rmessage> emptyBuffer(const std::vector<char>& data) {
      boost::intrusive::list<rmessage> retval;
      unsigned sz = 0;
      while (sz < data.size()) {
        uint32_t dest, psize;
        char* src = (char*)&dest;
        std::copy_n(&data[sz], sizeof(dest), src);
        src = (char*)&psize;
        sz += sizeof(dest);
        std::copy_n(&data[sz], sizeof(psize), src);
        sz += sizeof(psize);
        assert(psize);
        rmessage* im = new rmessage{dest, &data[sz], psize};
        retval.push_back(*im);
        sz += psize;
      }
      assert(sz == data.size());
      return retval;
    }

    bool empty() {
      return pending.empty();
    }

    std::chrono::high_resolution_clock::time_point oldest() {
      if (pending.empty())
        return std::chrono::high_resolution_clock::now();
      else
        return pending.front().time;
    }

    bool ready() {
      if (bytes == 0)
        return false;
      if (bytes > COMM_MIN) {
        //        std::cerr << "m ";
        return true;
      }
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - pending.front().time);
      if (elapsed.count() > COMM_DELAY) {
        //        std::cerr << elapsed.count() << " ";
        return true;
      }
      return false;
    }
  };

  std::vector<dataBuffer> sendData;
  dataBuffer recvData;
  std::atomic<int> quit;
  std::thread worker;

  bool isRouter() const {
    return ID == Num;
  }

  void isend(uint32_t dest, SendBuffer& buf) {
    uint32_t rdest = dest;
    if (!isRouter())
      rdest = Num;
    auto& sd = sendData[rdest];
    std::lock_guard<Galois::Runtime::LL::SimpleLock> lg(sd.lock);
    rmessage* m = new rmessage(dest, std::move(buf));
    sd.pending.push_back(*m);
    sd.bytes += sd.pending.back().size() + sizeof(uint32_t);
  }
  
  void trySendPacket(BaseIO& bio, uint32_t dest, dataBuffer& sd)
  {
    if (sd.ready() && bio.readySend(dest)) {
      std::lock_guard<LL::SimpleLock> lg(sd.lock);
      bool urg;
      auto d = sd.createBuffer(urg);
      assert(!d.empty());
      bio.send(typename BaseIO::message{dest, std::move(d), urg});
    }
  }

  void sendPacket(BaseIO& bio, uint32_t& bias) {
    if (!isRouter()) {
      trySendPacket(bio, Num, sendData[Num]);
    } else {
      for (uint32_t i = 0; i < Num; ++i) {
        uint32_t j = (i + bias) % Num;
        trySendPacket(bio, j, sendData[j]);
      }
      ++bias;
    }
  }

  void recvPacket(const typename BaseIO::message& m) {
    auto mesq = dataBuffer::emptyBuffer(m.data);
    if (isRouter()) { //don't need locks, only one thread
      while (!mesq.empty()) {
        auto& im = mesq.front();
        mesq.pop_front();
        auto& sd = sendData[im.dest];
        sd.pending.push_back(im);
        sd.bytes += sd.pending.back().size() + sizeof(uint32_t);
        sd.urgent |= m.urgent;
      }
    } else {
      std::lock_guard<LL::SimpleLock>(recvData.lock);
      recvData.pending.splice(recvData.pending.end(), mesq);
      //hack
      sendData[Num].urgent |= m.urgent;
    }
  }

  void workerThread() {
    { //block to hold bio
      BaseIO bio;
      Num = bio.Num() - 1;
      ID = bio.ID();
      decltype(sendData) v(Num+1);
      sendData.swap(v);
      quit = 0;
      uint32_t bias = 0;
      
      //loop
      while (!quit) {
        bio();
        if (bio.readyRecv()) //packets, woo!
          recvPacket(bio.recv());
        if (bio.readySend()) {
          sendPacket(bio, bias);
        }
      }
    } //destroy bio before acking quit
    quit = 2;
  }

public:
  using NetworkInterface::ID;
  using NetworkInterface::Num;

  NetworkInterfaceRouted() :quit(1), worker(&NetworkInterfaceRouted::workerThread, this) {
    while (quit) {}
  }

  virtual ~NetworkInterfaceRouted() {
    quit = 1;
    while (quit <= 1) {}
    worker.join();
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    assert(recv);
    assert(dest < Num);
    buf.serialize_header((void*)recv);
    isend(dest, buf);
  }

  virtual void flush() {
    assert(!isRouter());
    auto& sd = sendData[Num];
    sd.urgent = 1;
  }

  virtual bool handleReceives() {
    bool retval = false;
    recvData.lock.lock();
    if (!recvData.pending.empty()) {
      rmessage& m = recvData.pending.front();
      retval = !recvData.pending.empty();
      assert(m.buf.size());
      recvFuncTy f;
      uintptr_t fp;
      DeSerializeBuffer buf(std::move(m.buf));
      recvData.pending.pop_front();
      delete(&m);
      recvData.lock.unlock(); //FIXME: think about locking for message deliver order
      assert(buf.size());
      gDeserialize(buf,fp);
      assert(fp);
      f = (recvFuncTy)fp;
      //      std::cerr << "r";
      //Call deserialized function
      f(buf);
    } else {
      recvData.lock.unlock();
    }
    return retval;
  }
};

} //namespace ""

#ifdef USE_ROUTED_MPI
NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  static NetworkInterfaceRouted<NetworkIOMPI> net;
  while (net.ID == net.Num) {};
  return net;
}
#endif
