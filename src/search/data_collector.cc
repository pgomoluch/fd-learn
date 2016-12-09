#include "data_collector.h"

//#include "globals.h"
#include "evaluation_context.h"
#include "features.h"
#include "global_operator.h"
#include "state_registry.h"
#include "options/options.h"

using namespace std;

DataCollector::DataCollector() : ffh(Heuristic::default_options())
{

}

void DataCollector::test()
{
    cout << "DATA COLLECTOR\n";
    cout << g_operators.size() << endl;
}

void DataCollector::record_goal_path(SearchEngine *engine)
{
    const vector<const GlobalOperator *> &plan = engine->get_plan();
    StateRegistry *state_registry = engine->get_state_registry();
    // because state getters are not const
    //StateRegistry *u_state_registry = const_cast<StateRegistry*>(state_registry);
    StateRegistry *u_state_registry = state_registry;
    int plan_length = plan.size();
    int plan_cost = calculate_plan_cost(plan);
    
    GlobalState state = u_state_registry->get_initial_state(); // copying
    record_state(cout, state);
    record_data(cout, state, plan_cost, plan_length--);
    for(auto it = plan.begin(); it!=plan.end(); ++it)
    {
        state = u_state_registry->get_successor_state(state, **it); // copying
        plan_cost -= (*it)->get_cost();
        record_state(cout, state);
        record_data(cout, state, plan_cost, plan_length--);
    }
}

void DataCollector::record_state(ostream &out, const GlobalState &state)
{
    auto values = state.get_values();
    for(auto it=values.begin(); it!=values.end(); ++it)
    {
        out << *it << " ";
    }
    out << endl;
}

void DataCollector::record_data(ostream &out, const GlobalState &state, const int plan_cost, const int plan_length)
{
    // number of conjuncts in the goal
    out << g_goal.size() << " ";
    // Hamming distance to the goal
    out << features::distance(state) << " ";
    // number of applicable operators
    out << features::applicable_operator_count(state) << " ";
    // number of applicable operators which do not undo one of the goals
    out << features::non_diverging_operator_count(state) << " ";
    // FF heuristic
    EvaluationContext context(state);
    out << context.get_result(&ffh).get_h_value() << " ";
    
    // label: actual cost
    out << plan_cost << " ";
    // label: actual distance
    out << plan_length << endl;
}
