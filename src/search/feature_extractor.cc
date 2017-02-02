#include "globals.h"
#include "state_encoder.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char *argv[])
{
    cout << "Hello!\n";
    
    if(argc != 4)
    {
        cout << "Usage: feature_extractor <task MV representation file> <plan record file> <feature file>";
        return 1;
    }
    
    ifstream domain_file(argv[1]);
    read_everything(domain_file);
    domain_file.close();
    
    const unsigned state_length = g_variable_domain.size();
    
    ifstream plan_file(argv[2]);
    string goal_line;
    getline(plan_file, goal_line);
    stringstream goal_stream(goal_line);
    
    g_goal.clear();
    int var, val;
    while(goal_stream >> var, goal_stream >> val)
        g_goal.push_back(pair<int, int>(var, val));
    
    vector<int> values;
    int v;
    while(plan_file >> v)
        values.push_back(v);
    plan_file.close();
    
    StateEncoder state_encoder;
    ofstream features_file(argv[3]);
    for(unsigned i=0; i < values.size()/state_length; ++i)
    {
        vector<int> state(values.begin()+state_length*i, values.begin()+state_length*(i+1));
        StateRegistry state_registry(*g_root_task(), *g_state_packer, *g_axiom_evaluator,
        state);
        vector<double> features = state_encoder.encode(state_registry.get_initial_state());
        for(double f: features)
        {
            features_file << f << " "; 
        }
        features_file << endl;
    }
    features_file.close();
     
    return 0;
}
