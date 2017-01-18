#include "state_encoder.h"

#include "features.h"
#include "options/options.h"

using namespace std;

StateEncoder::StateEncoder() : ffh(Heuristic::default_options()), ceah(Heuristic::default_options())
{

}

vector<double> StateEncoder::encode(const GlobalState &state)
{
    vector<double> result;
    
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
    
    return result;
}

