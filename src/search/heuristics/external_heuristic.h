#ifndef HEURISTICS_EXTERNAL_HEURISTIC_H
#define HEURISTICS_EXTERNAL_HEURISTIC_H

#include <vector>

#include "../heuristic.h"
#include "../state_encoder.h"

namespace external_heuristic {

class ExternalHeuristic : public Heuristic {
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    ExternalHeuristic(const options::Options &options);
    ~ExternalHeuristic();
private:
    StateEncoder state_encoder;
    int fd;
};

}

#endif
