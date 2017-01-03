#include "learned_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

//#include "learned_evaluator.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace learned_heuristic {

LearnedHeuristic::LearnedHeuristic(const options::Options &options)
    : Heuristic(options) {
    cout << "Initializing learned heuristic..." << endl;
    ifstream in("../learned-heuristic/model.txt");
    if(!in)
        throw 42; // TMP
    double d;
    while(in >> d)
        model.push_back(d);
}

LearnedHeuristic::~LearnedHeuristic() {}

int LearnedHeuristic::compute_heuristic(const GlobalState &global_state) {
    auto features = data_collector.state_features(global_state);
    double result = 0.0;
    for(unsigned i=0; i<model.size(); ++i)
        result += model[i] * features[i];
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
        return new LearnedHeuristic(opts);
}

static Plugin<Heuristic> _plugin("learned", _parse);

}
