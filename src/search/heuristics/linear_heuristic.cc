#include "linear_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

//#include "learned_evaluator.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace linear_heuristic {

LinearHeuristic::LinearHeuristic(const options::Options &options)
    : Heuristic(options) {
    cout << "Initializing linear heuristic..." << endl;
    ifstream in("model.txt");
    if(!in)
        throw 42; // TMP
    in >> intercept;
    double d;
    while(in >> d)
        model.push_back(d);
}

LinearHeuristic::~LinearHeuristic() {}

int LinearHeuristic::compute_heuristic(const GlobalState &global_state) {
    auto features = state_encoder.encode(global_state);
    double result = intercept;
    for(unsigned i=0; i<model.size(); ++i)
        result += model[i] * features[i];
        
    if(result > 2000000000.0) // I would comapre to max(int), but wouldn't it bring precision issues?
        return 2147483647;
    return result;
}

static Heuristic *_parse(OptionParser &parser) {
    parser.document_synopsis("Learned heuristic", "");
    parser.document_language_support("action costs", "ignored by design");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support("axioms", "supported");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "no");
    parser.document_property("preferred operators", "no");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new LinearHeuristic(opts);
}

static Plugin<Heuristic> _plugin("linear", _parse);

}
