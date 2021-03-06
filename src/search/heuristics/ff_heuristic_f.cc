#include "ff_heuristic_f.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_tools.h"

#include <sstream>

using namespace std;

namespace ff_heuristic_f {

FFHeuristicF::FFHeuristicF(const Options &opts) : FFHeuristic(opts) {
    for(auto op: task_proxy.get_operators())
    {
        string name = get_schema_name(op);
        if (schema_map.count(name) == 0)
        {
            int old_size = schema_map.size();
            schema_map[name] = old_size;
        }
    }
    int schema_no = schema_map.size();
    for(int i = 0; i < schema_no; ++i)
    {
        pairwise_features.push_back(vector<bool>(schema_no, false));
    }
}

// The mark_preferred_operators_and_relaxed_plan and compute_heuristic functions
// are based on corresponding functions from the "pure" FF heuristic.
// We allowed massive code duplication to prevent any changes to the basic FF
// and thus enable a fair comparison.
void FFHeuristicF::mark_preferred_operators_and_relaxed_plan_f(
    const State &state, Proposition *goal, vector<UnaryOperator*> supported_ops,
    int depth) {
    if (!goal->marked) { // Only consider each subgoal once.
        goal->marked = true;
        UnaryOperator *unary_op = goal->reached_by;
        if (unary_op) { // We have not yet chained back to a start node.
            supported_ops.push_back(unary_op);
            for (size_t i = 0; i < unary_op->precondition.size(); ++i)
                mark_preferred_operators_and_relaxed_plan_f(
                    state, unary_op->precondition[i], supported_ops, depth+1);
            int operator_no = unary_op->operator_no;
            if (operator_no != -1) {
                // This is not an axiom.
                relaxed_plan[operator_no] = true;
                
                for (UnaryOperator *supported_op: supported_ops)
                {
                    for (Proposition *pre: supported_op->precondition)
                    {
                        if (unary_op->effect->id == pre->id)
                        {
                            int pred_id = schema_map[get_schema_name(task_proxy.get_operators()[unary_op->operator_no])];
                            int succ_id = schema_map[get_schema_name(task_proxy.get_operators()[supported_op->operator_no])];
                            pairwise_features[pred_id][succ_id] = true;
                        }
                    }
                }

                if (unary_op->cost == unary_op->base_cost) {
                    // This test is implied by the next but cheaper,
                    // so we perform it to save work.
                    // If we had no 0-cost operators and axioms to worry
                    // about, it would also imply applicability.
                    OperatorProxy op = task_proxy.get_operators()[operator_no];
                    if (is_applicable(op, state))
                        set_preferred(op);
                }
            }
        }
        else
        {
            if (depth > max_depth)
               max_depth = depth; 
        }
    }
}

int FFHeuristicF::compute_heuristic(const GlobalState &global_state) {
    
    unsigned n_features = features.size(); //TMP
    features.clear();
    dd_features.clear();
    int schema_no = schema_map.size();
    for(int i = 0; i < schema_no; ++i)
        for(int j = 0; j < schema_no; ++j)
            pairwise_features[i][j] = false;
    max_depth = 0;
    
    State state = convert_global_state(global_state);
    int h_add = compute_add_and_ff(state);
    if (h_add == DEAD_END)
    {
        features = vector<double>(n_features, 0.0);
        int n_schemata = schema_map.size();
        dd_features = vector<double>((n_schemata+1) * n_schemata, 0.0);
        return h_add;
    }

    // Collecting the relaxed plan also sets the preferred operators.
    for (size_t i = 0; i < goal_propositions.size(); ++i)
        mark_preferred_operators_and_relaxed_plan_f(state, goal_propositions[i]);

    int h_ff = 0;
    int operator_count = 0;
    int ignored_effect_count = 0;
    vector<int> schema_count(schema_map.size(), 0);
    for (size_t op_no = 0; op_no < relaxed_plan.size(); ++op_no) {
        if (relaxed_plan[op_no]) {
            relaxed_plan[op_no] = false; // Clean up for next computation.
            h_ff += task_proxy.get_operators()[op_no].get_cost();
            ++operator_count;
            string schema_name = get_schema_name(task_proxy.get_operators()[op_no]);
            schema_count[schema_map[schema_name]] += 1;
            ignored_effect_count += task_proxy.get_operators()[op_no].get_effects().size() - 1;
        }
    }
    
    features.push_back(operator_count);
    features.push_back(ignored_effect_count);
    
    double avg_ignored_effect_count = 0.0;
    if(operator_count > 0)
        avg_ignored_effect_count = (double)ignored_effect_count / operator_count;
    features.push_back(avg_ignored_effect_count);
    
    features.push_back((double)max_depth); // the number of layers in the graph
    
    // Domain-independent extract from schama count: quartiles
    vector<double> schema_count_copy(schema_count.begin(), schema_count.end());
    sort(schema_count_copy.begin(), schema_count_copy.end());
    for (int i = 1; i < 4; ++i)
    {
        int quartile = schema_count_copy[i * schema_count_copy.size() / 4];
        features.push_back((double)quartile);
    }
    
    // Domain-independent extract from the pairwise features matrix:
    // number of nonzero elements and symmetric pairs
    int nonzero_elements = 0;
    int symmetric_pairs = 0;
    int size = pairwise_features.size();
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            if (pairwise_features[i][j])
            {
                ++nonzero_elements;
                if (j >= i && pairwise_features[j][i])
                    ++symmetric_pairs;
            }
        }
    features.push_back((double)nonzero_elements);
    features.push_back((double)symmetric_pairs);
    
    dd_features.insert(dd_features.end(), schema_count.begin(), schema_count.end());
    for (auto row: pairwise_features)
        for (bool e: row)
            if (e)
                dd_features.push_back(1.0);
            else
                dd_features.push_back(0.0);
    
    return h_ff;
    //return operator_count;
    /*
    features.clear();
    
    int operator_count = 0;
    for(unsigned i=0; i<relaxed_plan.size(); ++i)
    {
        operator_count += relaxed_plan[i];
    }
    features.push_back(operator_count);
    
    return FFHeuristic::compute_heuristic(global_state);*/
}

string FFHeuristicF::get_schema_name(OperatorProxy op)
{
    stringstream stream(op.get_name());
    string base_name;
    stream >> base_name;
    return base_name;
}

static Heuristic *_parse(OptionParser &parser) {
    parser.document_synopsis("FF heuristic altered for feature extraction.", "See also Synergy.");
    parser.document_language_support("action costs", "supported");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support(
        "axioms",
        "supported (in the sense that the planner won't complain -- "
        "handling of axioms might be very stupid "
        "and even render the heuristic unsafe)");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "yes for tasks without axioms");
    parser.document_property("preferred operators", "yes");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new FFHeuristicF(opts);
}

static Plugin<Heuristic> _plugin("fff", _parse);

}
