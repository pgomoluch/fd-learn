#include "cea_heuristic_f.h"

#include "../domain_transition_graph.h"
#include "../global_operator.h"
#include "../global_state.h"
#include "../globals.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../task_tools.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <sstream>
#include <vector>

using namespace std;

/* Implementation notes:

   The main data structures are:
   - LocalProblem: a single "copy" of a domain transition graph, which
     is used to compute the costs of achieving all facts (v=d') for a
     fixed variable v starting from a fixed value d. So we can have at
     most |dom(v)| many local problems for any variable v. These are
     created lazily as needed.
   - LocalProblemNode: a single vertex in the domain transition graph
     represented by a LocalProblem. Knows what the successors in the
     graph are and keeps tracks of costs and helpful transitions for
     the node.
   - LocalTransition: a transition between two local problem nodes.
     Keeps track of how many unachieved preconditions there still are,
     what the cost of enabling the transition are and things like that.

   The following two design decisions might be worth revisiting:
   - Each local problem keeps its own copy of the graph itself
     (what is connected to what via which labels), even though this
     is not necessary. The "static" graph info and the "dynamic" info
     could be split, potentially saving quite a bit of memory.
   - The graph is encoded with reference cycles: each transition knows
     what its source node is, even though this is in a sense redundant
     (the source must be the node which holds the transition), and
     every node knows what its local problem is, which is similarly
     redundant (the local problem must be the one that holds this node).
     If we got rid of this, the main queue of the algorithm would need
     (LocalProblem *, LocalProblemNode *) pairs rather than straight
     node pointers, and the waiting lists would need to contain
     (LocalProblemNode *, LocalTransition *) pairs rather than straight
     transitions. So it's not clear if this would really save much, which
     is why we do not currently do it.
 */
namespace cea_heuristic_f {
struct LocalTransition {
    LocalProblemNode *source;
    LocalProblemNode *target;
    const ValueTransitionLabel *label;
    int action_cost;

    int target_cost;
    int unreached_conditions;

    LocalTransition(
        LocalProblemNode *source_, LocalProblemNode *target_,
        const ValueTransitionLabel *label_, int action_cost_)
        : source(source_), target(target_),
          label(label_), action_cost(action_cost_),
          target_cost(-1), unreached_conditions(-1) {
        // target_cost and unreached_cost are initialized by
        // expand_transition.
    }

    ~LocalTransition() {
    }
};


struct LocalProblemNode {
    // Attributes fixed during initialization.
    LocalProblem *owner;
    vector<LocalTransition> outgoing_transitions;

    // Dynamic attributes (modified during heuristic computation).
    int cost;
    bool expanded;
    vector<short> context;

    LocalTransition *reached_by;
    LocalTransition *first_on_path;

    vector<LocalTransition *> waiting_list;

    LocalProblemNode(LocalProblem *owner_, int context_size)
        : owner(owner_),
          cost(-1),
          expanded(false),
          context(context_size, -1),
          reached_by(0),
          first_on_path(0) {
    }

    ~LocalProblemNode() {
    }
};

struct LocalProblem {
    int base_priority;
    vector<LocalProblemNode> nodes;
    vector<int> *context_variables;
public:
    LocalProblem()
        : base_priority(-1) {
    }

    ~LocalProblem() {
    }
};

LocalProblem *ContextEnhancedAdditiveHeuristicF::get_local_problem(
    int var_no, int value) {
    LocalProblem * &table_entry = local_problem_index[var_no][value];
    if (!table_entry) {
        table_entry = build_problem_for_variable(var_no);
        local_problems.push_back(table_entry);
    }
    return table_entry;
}

LocalProblem *ContextEnhancedAdditiveHeuristicF::build_problem_for_variable(
    int var_no) const {
    LocalProblem *problem = new LocalProblem;

    DomainTransitionGraph *dtg = transition_graphs[var_no];

    problem->context_variables = &dtg->local_to_global_child;

    int num_parents = problem->context_variables->size();
    size_t num_values = task_proxy.get_variables()[var_no].get_domain_size();
    problem->nodes.reserve(num_values);
    for (size_t value = 0; value < num_values; ++value)
        problem->nodes.push_back(LocalProblemNode(problem, num_parents));

    // Compile the DTG arcs into LocalTransition objects.
    for (size_t value = 0; value < num_values; ++value) {
        LocalProblemNode &node = problem->nodes[value];
        const ValueNode &dtg_node = dtg->nodes[value];
        for (size_t i = 0; i < dtg_node.transitions.size(); ++i) {
            const ValueTransition &dtg_trans = dtg_node.transitions[i];
            int target_value = dtg_trans.target->value;
            LocalProblemNode &target = problem->nodes[target_value];
            for (const ValueTransitionLabel &label : dtg_trans.labels) {
                OperatorProxy op = label.is_axiom ?
                                   task_proxy.get_axioms()[label.op_id] :
                                   task_proxy.get_operators()[label.op_id];
                LocalTransition trans(&node, &target, &label, op.get_cost());
                node.outgoing_transitions.push_back(trans);
            }
        }
    }
    return problem;
}

LocalProblem *ContextEnhancedAdditiveHeuristicF::build_problem_for_goal() const {
    LocalProblem *problem = new LocalProblem;

    GoalsProxy goals_proxy = task_proxy.get_goals();

    problem->context_variables = new vector<int>;
    for (FactProxy goal : goals_proxy)
        problem->context_variables->push_back(goal.get_variable().get_id());

    for (size_t value = 0; value < 2; ++value)
        problem->nodes.push_back(LocalProblemNode(problem, goals_proxy.size()));

    vector<LocalAssignment> goals;
    for (size_t goal_no = 0; goal_no < goals_proxy.size(); ++goal_no) {
        int goal_value = goals_proxy[goal_no].get_value();
        goals.push_back(LocalAssignment(goal_no, goal_value));
    }
    vector<LocalAssignment> no_effects;
    ValueTransitionLabel *label = new ValueTransitionLabel(0, true, goals, no_effects);
    LocalTransition trans(&problem->nodes[0], &problem->nodes[1], label, 0);
    problem->nodes[0].outgoing_transitions.push_back(trans);
    return problem;
}

int ContextEnhancedAdditiveHeuristicF::get_priority(
    LocalProblemNode *node) const {
    /* Nodes have both a "cost" and a "priority", which are related.
       The cost is an estimate of how expensive it is to reach this
       node. The "priority" is the lowest cost value in the overall
       cost computation for which this node will be important. It is
       essentially the sum of the cost and a local-problem-specific
       "base priority", which depends on where this local problem is
       needed for the overall computation. */
    return node->owner->base_priority + node->cost;
}

inline void ContextEnhancedAdditiveHeuristicF::initialize_heap() {
    node_queue.clear();
}

inline void ContextEnhancedAdditiveHeuristicF::add_to_heap(
    LocalProblemNode *node) {
    node_queue.push(get_priority(node), node);
}

bool ContextEnhancedAdditiveHeuristicF::is_local_problem_set_up(
    const LocalProblem *problem) const {
    return problem->base_priority != -1;
}

void ContextEnhancedAdditiveHeuristicF::set_up_local_problem(
    LocalProblem *problem, int base_priority,
    int start_value, const State &state) {
    assert(problem->base_priority == -1);
    problem->base_priority = base_priority;

    for (auto &to_node : problem->nodes) {
        to_node.expanded = false;
        to_node.cost = numeric_limits<int>::max();
        to_node.waiting_list.clear();
        to_node.reached_by = 0;
    }

    LocalProblemNode *start = &problem->nodes[start_value];
    start->cost = 0;
    for (size_t i = 0; i < problem->context_variables->size(); ++i)
        start->context[i] = state[(*problem->context_variables)[i]].get_value();

    add_to_heap(start);
}

void ContextEnhancedAdditiveHeuristicF::try_to_fire_transition(
    LocalTransition *trans) {
    if (!trans->unreached_conditions) {
        LocalProblemNode *target = trans->target;
        if (trans->target_cost < target->cost) {
            target->cost = trans->target_cost;
            target->reached_by = trans;
            add_to_heap(target);
        }
    }
}

void ContextEnhancedAdditiveHeuristicF::expand_node(LocalProblemNode *node) {
    node->expanded = true;
    // Set context unless this was an initial node.
    LocalTransition *reached_by = node->reached_by;
    if (reached_by) {
        LocalProblemNode *parent = reached_by->source;
        vector<short> &context = node->context;
        context = parent->context;
        const vector<LocalAssignment> &precond = reached_by->label->precond;
        for (size_t i = 0; i < precond.size(); ++i)
            context[precond[i].local_var] = precond[i].value;
        const vector<LocalAssignment> &effect = reached_by->label->effect;
        for (size_t i = 0; i < effect.size(); ++i)
            context[effect[i].local_var] = effect[i].value;
        if (parent->reached_by) // if the parent isn't the beginning of the path
            node->first_on_path = parent->first_on_path;
        else
            node->first_on_path = node->reached_by;
    }
    for (size_t i = 0; i < node->waiting_list.size(); ++i) {
        LocalTransition *trans = node->waiting_list[i];
        assert(trans->unreached_conditions);
        --trans->unreached_conditions;
        trans->target_cost += node->cost;
        try_to_fire_transition(trans);
    }
    node->waiting_list.clear();
}

void ContextEnhancedAdditiveHeuristicF::expand_transition(
    LocalTransition *trans, const State &state) {
    /* Called when the source of trans is reached by Dijkstra
       exploration. Try to compute cost for the target of the
       transition from the source cost, action cost, and set-up costs
       for the conditions on the label. The latter may yet be unknown,
       in which case we "subscribe" to the waiting list of the node
       that will tell us the correct value. */

    assert(trans->source->cost >= 0);
    assert(trans->source->cost < numeric_limits<int>::max());

    trans->target_cost = trans->source->cost + trans->action_cost;

    if (trans->target->cost <= trans->target_cost) {
        // Transition cannot find a shorter path to target.
        return;
    }

    trans->unreached_conditions = 0;
    const vector<LocalAssignment> &precond = trans->label->precond;

    vector<LocalAssignment>::const_iterator
        curr_precond = precond.begin(),
        last_precond = precond.end();

    vector<short>::const_iterator context = trans->source->context.begin();
    vector<int>::const_iterator parent_vars =
        trans->source->owner->context_variables->begin();

    for (; curr_precond != last_precond; ++curr_precond) {
        int local_var = curr_precond->local_var;
        int current_val = context[local_var];
        int precond_value = curr_precond->value;
        int precond_var_no = parent_vars[local_var];

        if (current_val == precond_value)
            continue;

        LocalProblem *subproblem = get_local_problem(
            precond_var_no, current_val);

        if (!is_local_problem_set_up(subproblem)) {
            set_up_local_problem(
                subproblem, get_priority(trans->source), current_val, state);
        }

        LocalProblemNode *cond_node = &subproblem->nodes[precond_value];
        if (cond_node->expanded) {
            trans->target_cost += cond_node->cost;
            if (trans->target->cost <= trans->target_cost) {
                // Transition cannot find a shorter path to target.
                return;
            }
        } else {
            cond_node->waiting_list.push_back(trans);
            ++trans->unreached_conditions;
        }
    }
    try_to_fire_transition(trans);
}

int ContextEnhancedAdditiveHeuristicF::compute_costs(const State &state) {
    while (!node_queue.empty()) {
        pair<int, LocalProblemNode *> top_pair = node_queue.pop();
        int curr_priority = top_pair.first;
        LocalProblemNode *node = top_pair.second;

        assert(is_local_problem_set_up(node->owner));
        if (get_priority(node) < curr_priority)
            continue;
        if (node == goal_node)
            return node->cost;

        assert(get_priority(node) == curr_priority);
        expand_node(node);
        for (auto &transition : node->outgoing_transitions)
            expand_transition(&transition, state);
    }
    return DEAD_END;
}

void ContextEnhancedAdditiveHeuristicF::mark_helpful_transitions(
    LocalProblem *problem, LocalProblemNode *node, const State &state) {
    assert(node->cost >= 0 && node->cost < numeric_limits<int>::max());
    LocalTransition *first_on_path = node->first_on_path;
    if (first_on_path) {
        node->first_on_path = 0; // Clear to avoid revisiting this node later.
        if (first_on_path->target_cost == first_on_path->action_cost) {
            // Transition possibly applicable.
            const ValueTransitionLabel &label = *first_on_path->label;
            OperatorProxy op = label.is_axiom ?
                               task_proxy.get_axioms()[label.op_id] :
                               task_proxy.get_operators()[label.op_id];
            if (min_action_cost != 0 || is_applicable(op, state)) {
                // If there are no zero-cost actions, the target_cost/
                // action_cost test above already guarantees applicability.
                assert(!op.is_axiom());
                set_preferred(op);
            }
        } else {
            // Recursively compute helpful transitions for preconditions.
            int *context_vars = &*problem->context_variables->begin();
            for (const auto &assignment : first_on_path->label->precond) {
                int precond_value = assignment.value;
                int local_var = assignment.local_var;
                int precond_var_no = context_vars[local_var];
                if (state[precond_var_no].get_value() == precond_value)
                    continue;
                LocalProblem *subproblem = get_local_problem(
                    precond_var_no, state[precond_var_no].get_value());
                LocalProblemNode *subnode = &subproblem->nodes[precond_value];
                mark_helpful_transitions(subproblem, subnode, state);
            }
        }
    }
}

int ContextEnhancedAdditiveHeuristicF::compute_heuristic(const GlobalState &g_state) {
    const State state = convert_global_state(g_state);
    initialize_heap();
    goal_problem->base_priority = -1;
    for (LocalProblem *problem : local_problems)
        problem->base_priority = -1;

    set_up_local_problem(goal_problem, 0, 0, state);

    int heuristic = compute_costs(state);

    if (heuristic != DEAD_END && heuristic != 0)
        mark_helpful_transitions(goal_problem, goal_node, state);
    
    compute_features(goal_problem, goal_node, state);

    return heuristic;
}

ContextEnhancedAdditiveHeuristicF::ContextEnhancedAdditiveHeuristicF(
    const Options &opts)
    : Heuristic(opts),
      min_action_cost(get_min_operator_cost(task_proxy)) {
    cout << "Initializing context-enhanced additive heuristic..." << endl;

    DTGFactory factory(task_proxy, true, [](int, int) {return false; });
    transition_graphs = factory.build_dtgs();

    goal_problem = build_problem_for_goal();
    goal_node = &goal_problem->nodes[1];

    VariablesProxy vars = task_proxy.get_variables();
    local_problem_index.resize(vars.size());
    for (VariableProxy var : vars)
        local_problem_index[var.get_id()].resize(var.get_domain_size(), 0);
    
    // Initialize the structures for feature extraction
    
    // Schema dictionary
    for(auto op: task_proxy.get_operators())
    {
        string name = get_schema_name(op);
        if (schema_map.count(name) == 0)
        {
            int old_size = schema_map.size();
            schema_map[name] = old_size;
        }
    }
    
    // Schema occurrences
    int schema_no = schema_map.size();
    schema_count = vector<double>(schema_no);
    
    // Pairwise feature matrix
    for(int i = 0; i < schema_no; ++i)
    {
        pairwise_features.push_back(vector<bool>(schema_no, false));
    }
    
    // Print schema mapping
    for (auto const &e: schema_map)
        cout << e.first << "  " << e.second << endl;
}

ContextEnhancedAdditiveHeuristicF::~ContextEnhancedAdditiveHeuristicF() {
    if (goal_problem) {
        delete goal_problem->context_variables;
        delete goal_problem->nodes[0].outgoing_transitions[0].label;
    }
    delete goal_problem;

    for (LocalProblem *problem : local_problems)
        delete problem;
    for (DomainTransitionGraph *dtg : transition_graphs)
        delete dtg;
}

bool ContextEnhancedAdditiveHeuristicF::dead_ends_are_reliable() const {
    return false;
}

void ContextEnhancedAdditiveHeuristicF::compute_features(LocalProblem *problem, LocalProblemNode *node, const State &state) {
    
    features.clear();
    
    for (auto it = schema_count.begin(); it != schema_count.end(); ++it)
        *it = 0.0;
    
    for (vector<bool> &row: pairwise_features)
        for (auto it = row.begin(); it != row.end(); ++it)
            *it = false;
    
    max_graph_depth = 0;
    transition_count = 0;
    ignored_effect_count = 0;
    
    set_features_from_graph(problem, node, state, vector<OperatorProxy>(), 0);
    
    features.push_back((double)transition_count);
    features.push_back((double)ignored_effect_count);
    if (transition_count)
    {
        double avg_ignored_effect_count = (double)ignored_effect_count / transition_count;
        features.push_back(avg_ignored_effect_count);
    }
    else
    {
        features.push_back(0.0);
    }
    features.push_back((double)max_graph_depth);
}

void ContextEnhancedAdditiveHeuristicF::set_features_from_graph(LocalProblem *problem,
    LocalProblemNode *node, const State &state, vector<OperatorProxy> supported_ops, int depth) {
    
    if (depth > max_graph_depth)
        max_graph_depth = depth;
    LocalTransition *reached_by = node->reached_by;
    if (reached_by)
    {
        node->reached_by = 0;
        const ValueTransitionLabel &label = *reached_by->label;
        if(!label.is_axiom) {
            OperatorProxy op = task_proxy.get_operators()[label.op_id];
            string schema_name = get_schema_name(op);
            int schema_id = schema_map[schema_name];
            schema_count[schema_id] += 1.0;
            ++transition_count;
            ignored_effect_count += op.get_effects().size() - 1;
            
            // Update the pairwise features matrix
            for (OperatorProxy sop: supported_ops)
            {
                for (FactProxy pre: sop.get_preconditions())
                {
                    for (EffectProxy eff: op.get_effects())
                    {
                        FactProxy effect_fact = eff.get_fact();
                        if (pre == effect_fact)
                        {
                            int succ_id = schema_map[get_schema_name(sop)];
                            pairwise_features[schema_id][succ_id] = true;
                        }
                    }
                }
            }
            
            supported_ops.push_back(op);
        }
        if (reached_by->target_cost != reached_by->action_cost) // there are conditions
        {
            vector<int> *context_vars = problem->context_variables;
            for (const auto &assignment : reached_by->label->precond) {
                int precond_value = assignment.value;
                int local_var = assignment.local_var;
                int precond_var_no = (*context_vars)[local_var];
                if (state[precond_var_no].get_value() == precond_value)
                    continue;
                LocalProblem *subproblem = get_local_problem(
                    precond_var_no, state[precond_var_no].get_value());
                LocalProblemNode *subnode = &subproblem->nodes[precond_value];
                set_features_from_graph(subproblem, subnode, state, supported_ops, ++depth);
            }
        }
    } 
}

vector<double> ContextEnhancedAdditiveHeuristicF::get_dd_features()
{
    vector<double> result(schema_count.begin(), schema_count.end());
    
    for (auto row: pairwise_features)
        for (bool e: row)
            if(e)
                result.push_back(1.0);
            else
                result.push_back(0.0);
    
    return result;
}

string ContextEnhancedAdditiveHeuristicF::get_schema_name(const OperatorProxy &op) {
    stringstream stream(op.get_name());
    string base_name;
    stream >> base_name;
    return base_name;
}

static Heuristic *_parse(OptionParser &parser) {
    parser.document_synopsis("Context-enhanced additive heuristic with learning features", "");
    parser.document_language_support("action costs", "supported");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support(
        "axioms",
        "supported (in the sense that the planner won't complain -- "
        "handling of axioms might be very stupid "
        "and even render the heuristic unsafe)");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "no");
    parser.document_property("preferred operators", "yes");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();

    if (parser.dry_run())
        return 0;
    else
        return new ContextEnhancedAdditiveHeuristicF(opts);
}

static Plugin<Heuristic> _plugin("ceaf", _parse);
}
