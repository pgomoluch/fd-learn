#ifndef STATE_ENCODER_H
#define STATE_ENCODER_H

#include <vector>

#include "evaluation_context.h"
#include "global_state.h"
#include "heuristics/cea_heuristic.h"
#include "heuristics/ff_heuristic_f.h"

class StateEncoder
{
    const unsigned N_DOMAIN_QUANTILES = 4;
public:
    StateEncoder();
    std::vector<double> encode(const GlobalState &state);
private:
    ff_heuristic::FFHeuristicF ffh;
    cea_heuristic::ContextEnhancedAdditiveHeuristic ceah;
    std::vector<int> domain_sizes;
    std::vector<int> domain_quantiles;
};

#endif
