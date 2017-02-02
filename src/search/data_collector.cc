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
    
    ofstream state_stream("states.txt");
    ofstream feature_stream("features.txt");
    ofstream label_stream("labels.txt");
    
    // Add goal specification to the states record
    for(auto entry: g_goal)
        state_stream << entry.first << " " << entry.second << " ";
    state_stream << endl;
    
    GlobalState state = state_registry->get_initial_state(); // copying
    record_state(state_stream, state);
    record_data(feature_stream, label_stream, state, plan_cost, plan_length--);
    for(auto it = plan.begin(); it!=plan.end(); ++it)
    {
        state = state_registry->get_successor_state(state, **it); // copying
        plan_cost -= (*it)->get_cost();
        record_state(state_stream, state);
        record_data(feature_stream, label_stream, state, plan_cost, plan_length--);
    }
    
    state_stream.close();
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
    vector<double> features = state_encoder.encode(state);
    for(double f: features)
        fs << f << " ";

    if(&fs == &ls)
        fs << " ";
    else
        fs << endl;
    
    // label: actual cost
    ls << plan_cost << endl;
    // label: actual distance
    //out << plan_length << endl;
    if(plan_length < -1081*2) ls << "#"; // you know why
}

