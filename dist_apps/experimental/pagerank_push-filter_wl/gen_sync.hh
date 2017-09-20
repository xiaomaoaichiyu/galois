#include "galois/Runtime/sync_structures.h"

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(nout, unsigned int);
GALOIS_SYNC_STRUCTURE_BROADCAST(nout, unsigned int);
GALOIS_SYNC_STRUCTURE_BITSET(nout);

GALOIS_SYNC_STRUCTURE_REDUCE_SET_ARRAY(residual, float);
GALOIS_SYNC_STRUCTURE_REDUCE_ADD_ARRAY(residual, float);
GALOIS_SYNC_STRUCTURE_BROADCAST_ARRAY(residual, float);
GALOIS_SYNC_STRUCTURE_BITSET(residual);
