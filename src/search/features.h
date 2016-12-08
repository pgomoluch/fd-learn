#ifndef FEATURES_H
#define FEATURES_H

#include "global_state.h"

namespace features
{
    int distance(const GlobalState &state);
    int applicable_operator_count(const GlobalState &state);
    int non_diverging_operator_count(const GlobalState &state);
    
    bool diverges_from_goal(const GlobalOperator &op, const GlobalState &state);
    bool defines_goal(int var);
    bool conjunct_satisfied(int var, const GlobalState &state);
}

#endif
