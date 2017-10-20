#ifndef STATE_ENCODER_H
#define STATE_ENCODER_H

#include <vector>

#include "evaluation_context.h"
#include "global_state.h"
#include "heuristics/cea_heuristic_f.h"
#include "heuristics/ff_heuristic_f.h"

class StateEncoder
{
    const unsigned N_DOMAIN_QUANTILES = 4;
public:
    StateEncoder();
    std::vector<double> encode(const GlobalState &state);
    const vector<const GlobalOperator *> &get_preferred_operators();
private:
    ff_heuristic::FFHeuristicF ffh;
    cea_heuristic_f::ContextEnhancedAdditiveHeuristicF ceah;
    std::vector<int> domain_sizes;
    std::vector<int> domain_quantiles;
    std::vector<const GlobalOperator*> preferred_operators;
    
    static int distance(const GlobalState &state);
    static int applicable_operator_count(const GlobalState &state);
    static int non_diverging_operator_count(const GlobalState &state);
    static bool diverges_from_goal(const GlobalOperator &op, const GlobalState &state);
    static bool defines_goal(int var);
    static bool conjunct_satisfied(int var, const GlobalState &state);
};

#endif
