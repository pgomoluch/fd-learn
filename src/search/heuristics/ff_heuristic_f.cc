#include "ff_heuristic_f.h"

#include "../option_parser.h"

namespace ff_heuristic {

FFHeuristicF::FFHeuristicF(const Options &opts) : FFHeuristic(opts) {
}

int FFHeuristicF::compute_heuristic(const GlobalState &global_state) {
    
    features.clear();
    
    State state = convert_global_state(global_state);
    int h_add = compute_add_and_ff(state);
    if (h_add == DEAD_END)
        return h_add;

    // Collecting the relaxed plan also sets the preferred operators.
    for (size_t i = 0; i < goal_propositions.size(); ++i)
        mark_preferred_operators_and_relaxed_plan(state, goal_propositions[i]);

    int h_ff = 0;
    int operator_count = 0;
    int ignored_effect_count = 0;
    for (size_t op_no = 0; op_no < relaxed_plan.size(); ++op_no) {
        if (relaxed_plan[op_no]) {
            relaxed_plan[op_no] = false; // Clean up for next computation.
            h_ff += task_proxy.get_operators()[op_no].get_cost();
            ++operator_count;
            ignored_effect_count += task_proxy.get_operators()[op_no].get_effects().size() - 1;
        }
    }
    
    features.push_back(operator_count);
    features.push_back(ignored_effect_count);
    
    double avg_ignored_effect_count = 0.0;
    if(operator_count > 0)
        avg_ignored_effect_count = (double)ignored_effect_count / operator_count;
    features.push_back(avg_ignored_effect_count);
    
    return h_ff;    
    /*
    features.clear();
    
    int operator_count = 0;
    for(unsigned i=0; i<relaxed_plan.size(); ++i)
    {
        operator_count += relaxed_plan[i];
    }
    features.push_back(operator_count);
    
    return FFHeuristic::compute_heuristic(global_state);*/
}

}
