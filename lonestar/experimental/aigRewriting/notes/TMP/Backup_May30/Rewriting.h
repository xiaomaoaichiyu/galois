#ifndef RECONVDRIVENWINDOWING_RECONVDRIVENWINDOWING_H_
#define RECONVDRIVENWINDOWING_RECONVDRIVENWINDOWING_H_

#include "Aig.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Timer.h"

namespace algorithm {

class ReconvDrivenWindowing {

private:

	aig::Aig & aig;

public:

	static int windowCounter;

	ReconvDrivenWindowing( aig::Aig & aig );
	virtual ~ReconvDrivenWindowing();

	void run( int nOutputs, int nInputs, int nLevels, int nFanout, int cutSizeLimit, int verbose );
};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* RECONVDRIVENWINDOWING_H_ */
