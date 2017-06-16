#include "neural_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace neural_heuristic {

NeuralHeuristic::NeuralHeuristic(const options::Options &options)
    : Heuristic(options), network("network.txt") {
    cout << "Initializing neural heuristic..." << endl;
}

int NeuralHeuristic::compute_heuristic(const GlobalState &global_state) {
    auto features = state_encoder.encode(global_state);
    double result = network.evaluate(features);
    
    //if(result > 2000000000.0)
    //    return 2147483647;
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
        return new NeuralHeuristic(opts);
}

static Plugin<Heuristic> _plugin("neural", _parse);

}
