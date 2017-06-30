#include "state_encoder.h"

//#include "features.h"
#include "global_operator.h"
#include "globals.h"
#include "options/options.h"

using namespace std;

StateEncoder::StateEncoder() : ffh(Heuristic::default_options()), ceah(Heuristic::default_options())
{
    domain_sizes = g_variable_domain;
    sort(domain_sizes.begin(), domain_sizes.end());
    domain_quantiles.reserve(N_DOMAIN_QUANTILES);
    for(unsigned i=0; i < N_DOMAIN_QUANTILES; ++i)
    {
        unsigned id = i * domain_sizes.size() / N_DOMAIN_QUANTILES;
        domain_quantiles.push_back(domain_sizes[id]);
    }
}

vector<double> StateEncoder::encode(const GlobalState &state)
{
    vector<double> result;
    
    // number of variables
    result.push_back(domain_sizes.size());
    // variable domain size distribution
    result.insert(result.end(), domain_quantiles.begin(), domain_quantiles.end());
    // number of conjuncts in the goal
    result.push_back(g_goal.size());
    // Hamming distance to the goal
    result.push_back(distance(state));
    // number of applicable operators
    result.push_back(applicable_operator_count(state));
    // number of applicable operators which do not undo one of the goals
    result.push_back(non_diverging_operator_count(state));
    
    EvaluationContext context(state);
    // FF heuristic
    result.push_back(context.get_result(&ffh).get_h_value());
    // CEA heuristic
    result.push_back(context.get_result(&ceah).get_h_value());
    
    // FF derived features
    vector<double> ff_features = ffh.get_features();
    result.insert(result.end(), ff_features.begin(), ff_features.end());
    
    // Domain-dependent FF derived features
    vector<double> ff_dd_features = ffh.get_dd_features();
    result.insert(result.end(), ff_dd_features.begin(), ff_dd_features.end());
    
    return result;
}

int StateEncoder::distance(const GlobalState &state)
{
    int distance = 0;
    for (size_t i = 0; i < g_goal.size(); ++i)
        if(state.get_values()[g_goal[i].first] != g_goal[i].second)
            ++distance;
    
    return distance;
}

int StateEncoder::applicable_operator_count(const GlobalState &state)
{
    int count = 0;
    for(auto it = g_operators.begin(); it != g_operators.end(); ++it)
    {
        if(it->is_applicable(state))
            ++count;
    }
    return count;
}

int StateEncoder::non_diverging_operator_count(const GlobalState &state)
{
    int count = 0;
    for(auto it = g_operators.begin(); it != g_operators.end(); ++it)
    {
        if(it->is_applicable(state))
            if(!diverges_from_goal(*it, state))
                count++;
    }
    return count;
}
   
bool StateEncoder::diverges_from_goal(const GlobalOperator &op, const GlobalState &state)
{
    for(auto it = op.get_effects().begin(); it != op.get_effects().end(); ++it)
    {
        if(it->does_fire(state) && defines_goal(it->var))
            if(conjunct_satisfied(it->var, state) && state[it->var] != it->val)
                return true; 
    }
    return false;
}

bool StateEncoder::defines_goal(int var)
{
    for(size_t i = 0; i < g_goal.size(); ++i)
        if(g_goal[i].first == var)
            return true;
    return false;
}

bool StateEncoder::conjunct_satisfied(int var, const GlobalState &state)
{
     for(size_t i = 0; i < g_goal.size(); ++i)
        if(g_goal[i].first == var)
            return state[i] == g_goal[i].second;
    return true;
}
