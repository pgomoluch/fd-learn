#include "data_collector.h"

#include <fstream>
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
    int plan_length = plan.size();
    int plan_cost = calculate_plan_cost(plan);
    
    ofstream feature_stream("features.txt");
    ofstream label_stream("labels.txt");
    
    GlobalState state = state_registry->get_initial_state(); // copying
    record_state(cout, state);
    record_data(feature_stream, label_stream, state, plan_cost, plan_length--);
    for(auto it = plan.begin(); it!=plan.end(); ++it)
    {
        state = state_registry->get_successor_state(state, **it); // copying
        plan_cost -= (*it)->get_cost();
        record_state(cout, state);
        record_data(feature_stream, label_stream, state, plan_cost, plan_length--);
    }
    
    feature_stream.close();
    label_stream.close();
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

void DataCollector::record_data(ostream &fs, ostream &ls, const GlobalState &state, const int plan_cost, const int plan_length)
{
    // number of conjuncts in the goal
    fs << g_goal.size() << " ";
    // Hamming distance to the goal
    fs << features::distance(state) << " ";
    // number of applicable operators
    fs << features::applicable_operator_count(state) << " ";
    // number of applicable operators which do not undo one of the goals
    fs << features::non_diverging_operator_count(state) << " ";
    // FF heuristic
    EvaluationContext context(state);
    fs << context.get_result(&ffh).get_h_value();
    if(fs == ls)
        fs << " ";
    else
        fs << endl;
    
    // label: actual cost
    ls << plan_cost << endl;
    // label: actual distance
    //out << plan_length << endl;
    if(plan_length < -1081*2) ls << "#"; // you know why
}

