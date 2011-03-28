// worklists building blocks -*- C++ -*-

#ifndef __WORKLISTHELPERS_H_
#define __WORKLISTHELPERS_H_

namespace GaloisRuntime {
namespace WorkList {

template<typename T, int chunksize = 64, bool concurrent = true>
class FixedSizeRing :private boost::noncopyable, private PaddedLock<concurrent> {
  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::unlock;
  unsigned start;
  unsigned end;
  T data[chunksize];

  bool _i_empty() {
    return start == end;
  }

  bool _i_full() {
    return (end + 1) % chunksize == start;
  }

public:
  
  template<bool newconcurrent>
  struct rethread {
    typedef FixedSizeRing<T, chunksize, newconcurrent> WL;
  };

  typedef T value_type;

  FixedSizeRing() :start(0), end(0) {}

  bool empty() {
    lock();
    bool retval = _i_empty();
    unlock();
    return retval;
  }

  bool full() {
    lock();
    bool retval = _i_full();
    unlock();
    return retval;
  }

  bool push_front(value_type val) {
    lock();
    if (_i_full()) {
      unlock();
      return false;
    }
    start += chunksize - 1;
    start %= chunksize;
    data[start] = val;
    unlock();
    return true;
  }

  bool push_back(value_type val) {
    lock();
    if (_i_full()) {
      unlock();
      return false;
    }
    data[end] = val;
    end += 1;
    end %= chunksize;
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop_front() {
    lock();
    if (_i_empty()) {
      unlock();
      return std::make_pair(false, value_type());
    }
    value_type retval = data[start];
    ++start;
    start %= chunksize;
    unlock();
    return std::make_pair(true, retval);
  }

  std::pair<bool, value_type> pop_back() {
    lock();
    if (_i_empty()) {
      unlock();
      return std::make_pair(false, value_type());
    }
    end += chunksize - 1;
    end %= chunksize;
    value_type retval = data[end];
    unlock();
    return std::make_pair(true, retval);
  }
};




}
}


#endif
