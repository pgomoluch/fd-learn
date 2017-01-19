#include "globals.h"
#include "state_registry.h"
#include "successor_generator.h"
#include "utils/rng.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

int main(int argc, char *argv[])
{   
    if(argc != 2)
    {
        cout << "Usage: random_walker <number of steps>" << endl;
        return 1;
    }
    
    const int num_steps = atoi(argv[1]);
    
    read_everything(cin);
    cout << "Read everything.\n";
    
    // Create a random goal state
    g_rng()->seed(time(0));
    const unsigned state_length = g_variable_domain.size();
    vector<int> goal_state(state_length, -1);
    for(auto v: g_goal)
        goal_state[v.first] = v.second;
    for(unsigned i=0; i<state_length; ++i)
        if(goal_state[i] == -1)
            goal_state[i] = (*g_rng())(g_variable_domain[i]);
    
    // Evaluate the axioms to ensure consistency
    int num_bins = g_state_packer->get_num_bins();
    IntPacker::Bin *buffer = new IntPacker::Bin[num_bins];
    for(unsigned i=0; i<state_length; ++i)
        g_state_packer->set(buffer, i , goal_state[i]);
    g_axiom_evaluator->evaluate(buffer, *g_state_packer);
    for(unsigned i=0; i<g_variable_domain.size(); ++i)
        goal_state[i] = g_state_packer->get(buffer, i);
    
    StateRegistry state_registry(*g_root_task(), *g_state_packer, *g_axiom_evaluator,
        goal_state);
    GlobalState state = state_registry.get_initial_state();
    
    for(int i=0; i<num_steps; ++i)
    {
        vector<const GlobalOperator *> applicable_operators;
        g_successor_generator->generate_applicable_ops(state, applicable_operators);
        state = state_registry.get_successor_state(state,
            *applicable_operators[(*g_rng())(applicable_operators.size())]);
    }
    
    auto values = state.get_values();
    cout << "begin_state" << endl;
    for(auto it=values.begin(); it!=values.end(); ++it)
        cout << *it << endl;
    cout << "end_state" << endl;
    
    delete[] buffer;
    return 0;
}
