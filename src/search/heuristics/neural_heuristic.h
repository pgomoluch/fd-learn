#ifndef HEURISTICS_NEURAL_HEURISTIC_H
#define HEURISTICS_NEURAL_HEURISTIC_H

#include <vector>

#include "../heuristic.h"
#include "../state_encoder.h"
#include "../nn/network.h"

namespace neural_heuristic {

class NeuralHeuristic : public Heuristic {
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    NeuralHeuristic(const options::Options &options);
private:
    StateEncoder state_encoder;
    Network network;
};

}

#endif
