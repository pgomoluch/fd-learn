#ifndef STATE_ENCODER_H
#define STATE_ENCODER_H

#include <vector>

#include "evaluation_context.h"
#include "global_state.h"
#include "heuristics/ff_heuristic.h"

class StateEncoder
{
public:
    StateEncoder();
    std::vector<double> encode(const GlobalState &state);
private:
    ff_heuristic::FFHeuristic ffh;
};

#endif
