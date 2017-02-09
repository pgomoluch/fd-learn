#include "state_encoder.h"

#include "features.h"
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
    result.push_back(features::distance(state));
    // number of applicable operators
    result.push_back(features::applicable_operator_count(state));
    // number of applicable operators which do not undo one of the goals
    result.push_back(features::non_diverging_operator_count(state));
    
    EvaluationContext context(state);
    // FF heuristic
    result.push_back(context.get_result(&ffh).get_h_value());
    // CEA heuristic
    result.push_back(context.get_result(&ceah).get_h_value());
    
    // FF derived features
    vector<double> ff_features = ffh.get_features();
    result.insert(result.end(), ff_features.begin(), ff_features.end());
    
    return result;
}

