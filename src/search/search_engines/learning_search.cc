#include "learning_search.h"

#include "search_common.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"
#include "../successor_generator.h"

#include "../open_lists/ra_alternation_open_list.h"
#include "../utils/memory.h"

#include <iomanip>

using namespace std;
using namespace std::chrono;

namespace learning_search {

LearningSearch::LearningSearch(const Options &opts)
    : SearchEngine(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      open_list_factory(opts.get<shared_ptr<RAOpenListFactory>>("ra_open")),
      global_open_list(open_list_factory->create_state_open_list()),
      local_open_list(nullptr),
      f_evaluator(opts.get<ScalarEvaluator *>("f_eval", nullptr)),
      preferred_operator_heuristics(opts.get_list<Heuristic *>("preferred")),
      learning_rate(opts.get<double>("learning_rate")),
      rng(system_clock::now().time_since_epoch().count()),
      //rng(0),
      learning_log("rl-log.txt"),
      real_dist(0.0, 1.0),
      int_dist(0, actions.size()-1) {
    
    Options state_list_opts;
    const vector<ScalarEvaluator*> &evals =
            opts.get_list<ScalarEvaluator *>("evals");
    
    open_list = global_open_list.get();
}

void LearningSearch::initialize() {
    cout << "Conducting learning search..." << endl;
    cout << "The learning rate is " << learning_rate << endl;
    assert(open_list);

    set<Heuristic*> hset;
    open_list->get_involved_heuristics(hset);

    hset.insert(preferred_operator_heuristics.begin(),
        preferred_operator_heuristics.end());

    heuristics.assign(hset.begin(), hset.end());
    assert(!heuristics.empty());

    ifstream weight_file("weights.txt");
    if(weight_file) {
        for (double &w: weights)
            weight_file >> w;
        weight_file.close();
    }


    const GlobalState &initial_state = state_registry.get_initial_state();
    for (Heuristic *h: heuristics) {
        h->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);

    statistics.inc_evaluated_states();

    if (open_list->is_dead_end(eval_context)) {
        cout << "The initial state is a dead end." << endl;
    } else {
        //start_f_value_statistics(eval_context);
        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();

        // At the moment we only support a single heuristic
        int h = eval_context.get_heuristic_value(heuristics[0]);
        all_time_best_h = best_h = previous_best_h = h;
        open_list->insert(eval_context, initial_state.get_id());
    }
}

void LearningSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

SearchStatus LearningSearch::step() {
    steady_clock::time_point now = steady_clock::now();
    //if (step_counter % 100 == 0)
    if (duration_cast<milliseconds>(now - action_start).count() >= ACTION_DURATION)
        update_routine();
    ++step_counter;
    return (this->*actions[current_action_id])();
}

SearchStatus LearningSearch::greedy_step() {
    return simple_step(false);
}

SearchStatus LearningSearch::epsilon_greedy_step() {
    return simple_step(true);
}

SearchStatus LearningSearch::simple_step(bool randomized) {

    pair<SearchNode, bool> n = fetch_next_node(randomized);
    if (!n.second) {
        return FAILED;
    }
    SearchNode node = n.first;

    GlobalState state = node.get_state();
    if (check_goal_and_set_plan(state)) {
        terminate_learning();
        return SOLVED;
    }

    vector<const GlobalOperator*> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);

    // Like in EagerSearch, this evaluates the expanded state (again) to get preferred ops.
    EvaluationContext eval_context(state, node.get_g(), false, &statistics, true);
    algorithms::OrderedSet<const GlobalOperator *> preferred_operators =
        collect_preferred_operators(eval_context, preferred_operator_heuristics);

    ++expansions_without_progress;
    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        bool is_preferred = preferred_operators.contains(op);
        process_state(node, state, op, succ_state, is_preferred);
    }

    return IN_PROGRESS;
}

SearchStatus LearningSearch::rollout_step() {

    pair<SearchNode, bool> n = fetch_next_node(false);
    if (!n.second) {
        return FAILED;
    }
    SearchNode node = n.first;

    GlobalState state = node.get_state();
    if (check_goal_and_set_plan(state)) {
        terminate_learning();
        return SOLVED;
    }
    
    vector<const GlobalOperator*> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);
    
    // Fully expand the processed state

    // Get preferred operators
    EvaluationContext eval_context(state, node.get_g(), false, &statistics, true);
    algorithms::OrderedSet<const GlobalOperator *> preferred_operators =
        collect_preferred_operators(eval_context, preferred_operator_heuristics);
    
    ++expansions_without_progress;
    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        bool is_preferred = preferred_operators.contains(op);
        process_state(node, state, op, succ_state, is_preferred);
    }

    if (applicable_ops.size() == 0 || expansions_without_progress < STALL_SIZE)
        return IN_PROGRESS;
    
    // Perform a stochastic rollout from expanded state
    const GlobalOperator *op = applicable_ops[rng() % applicable_ops.size()];
    GlobalState rollout_state = state_registry.get_successor_state(state, *op);
    for (unsigned i = 0; i < ROLLOUT_LENGTH; ++i) {
        vector<const GlobalOperator*> applicable_ops;
        g_successor_generator->generate_applicable_ops(rollout_state, applicable_ops);
        if (applicable_ops.size() == 0)
            break;
        const GlobalOperator *op = applicable_ops[rng() % applicable_ops.size()];
        GlobalState succ_state = state_registry.get_successor_state(rollout_state, *op);
        statistics.inc_generated();
        SearchNode node = search_space.get_node(rollout_state);
        learning_log << "#";
        // Get preferred operators for the state
        EvaluationContext eval_context(state, node.get_g(), false, &statistics, true);
        preferred_operators =
            collect_preferred_operators(eval_context, preferred_operator_heuristics);
        
        bool is_preferred = preferred_operators.contains(op);
        process_state(node, rollout_state, op, succ_state, is_preferred);
        if(expansions_without_progress == 0)
            break;
        rollout_state = succ_state;
    }

    learning_log << endl;
    return IN_PROGRESS;
}

SearchStatus LearningSearch::preferred_rollout_step() {

    pair<SearchNode, bool> n = fetch_next_node(false);
    if (!n.second) {
        return FAILED;
    }
    SearchNode node = n.first;

    GlobalState state = node.get_state();
    if (check_goal_and_set_plan(state)) {
        terminate_learning();
        return SOLVED;
    }
        
    
    vector<const GlobalOperator*> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);
    
    // Fully expand the processed state

    // Get preferred operators
    EvaluationContext eval_context(state, node.get_g(), false, &statistics, true);
    algorithms::OrderedSet<const GlobalOperator *> preferred_operators =
        collect_preferred_operators(eval_context, preferred_operator_heuristics);

    ++expansions_without_progress;
    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        bool is_preferred = preferred_operators.contains(op);
        process_state(node, state, op, succ_state, is_preferred);
    }

    if (applicable_ops.size() == 0 || expansions_without_progress < STALL_SIZE)
        return IN_PROGRESS;
    
    preferred_operators.shuffle(*g_rng());

    // Perform a preferred operators rollout from the expanded state
    GlobalState rollout_state = state;
    for (unsigned i = 0; i < ROLLOUT_LENGTH; ++i) {

        // Find an applicable preferred operator
        std::vector<const GlobalOperator *>::const_iterator it;
        for (it = preferred_operators.begin(); it != preferred_operators.end(); it++) {
            if ((*it)->is_applicable(rollout_state)) {
                break;
            }
        }
        if (it == preferred_operators.end()) // no applicable operators
            break;

        GlobalState succ_state = state_registry.get_successor_state(rollout_state, **it);
        statistics.inc_generated();
        SearchNode node = search_space.get_node(rollout_state);
        learning_log << "#";
        process_state(node, rollout_state, *it, succ_state, true);
        if (expansions_without_progress == 0)
            break;
        
        EvaluationContext eval_context(succ_state, node.get_g()+(*it)->get_cost(), false, &statistics, true);
        preferred_operators = collect_preferred_operators(eval_context, heuristics);
        preferred_operators.shuffle(*g_rng());

        rollout_state = succ_state;
    }
    learning_log << endl;
    return IN_PROGRESS;
}

SearchStatus LearningSearch::local_step() {
    if (!local_open_list) {
        // Create a new local queue and switch
        if (open_list->empty())
            return FAILED;

        StateID id = get_best_state();
        GlobalState state = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(state);

        local_open_list = open_list_factory->create_state_open_list();
        EvaluationContext eval_context(state, node.get_g(), true, &statistics);
        local_open_list->insert(eval_context, id);
        
        open_list = local_open_list.get();
    }
    SearchStatus status = simple_step(false);
    // When local search fails, try to get another state from the global queue
    if (status == FAILED) {
        if (global_open_list->empty())
            return FAILED;
        
        StateID id = global_open_list->remove_min();
        GlobalState state = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(state);
        EvaluationContext eval_context(state, node.get_g(), true, &statistics);
        local_open_list->insert(eval_context, id);
        status = IN_PROGRESS;
    }
    return status;
}

void LearningSearch::merge_local_list() {
    while (!local_open_list->empty()) {
        StateID id = local_open_list->remove_min();
        GlobalState state = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(state);
        // None of the local search states are preferred
        EvaluationContext eval_context(state, node.get_g(), false, &statistics);
        global_open_list->insert(eval_context, id);
    }
}

pair<SearchNode, bool> LearningSearch::fetch_next_node(bool randomized) {
    while (true) {
        if (open_list->empty()) {
            cout << "State queue is empty -- no solution!" << endl;
            // HACK after eager_search.cc, because there's no default constructor
            // for SearchNode
            const GlobalState &initial_state = state_registry.get_initial_state();
            SearchNode dummy_node = search_space.get_node(initial_state);
            return make_pair(dummy_node, false);
        }

        StateID id = StateID::no_state;
        if (randomized)
            id = get_randomized_state();
        else
            id = get_best_state();
        
        //StateID _id = (this->*actions[current_action_id])();
        GlobalState state = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(state);

        if (node.is_closed())
            continue;

        node.close();
        assert(!node.is_dead_end());
        //update_f_value_statistics(node);
        statistics.inc_expanded(); // why here?
        return make_pair(node, true);
    }
}

StateID LearningSearch::get_best_state() {
    return open_list->remove_min();
}

StateID LearningSearch::get_randomized_state() {
    return open_list->remove_epsilon();
}

void LearningSearch::update_routine() {
    // Merge the local list if any and switch to the global one.
    if (local_open_list) {
        merge_local_list();
        local_open_list = nullptr;
        open_list = global_open_list.get();
    }
    
    int reward = 0;
    if (step_counter > 0) {
        reward = previous_best_h - best_h;
        previous_best_h = best_h;
        
        weights[current_action_id] =
            (1 - learning_rate) * weights[current_action_id]
            + learning_rate * (reward > 0);
    }
    
    current_action_id = epsilon_greedy_policy();
    //current_action_id = proportional_policy();

    // Dev logging
    steady_clock::time_point now = steady_clock::now();
    if (step_counter > 0) {
        cout << "Reward: " << reward;
        cout << ", Duration: "
            << duration_cast<milliseconds>(now - action_start).count();
        cout << ", Steps: " << step_counter - steps_at_action_start << endl;
    }
    cout << "Weights: " << setprecision(3);
    for(double d: weights)
        cout << d << " ";
    cout << ", Choice: " << current_action_id << ", ";
    cout << "ENP: " << expansions_without_progress << ", ";

    action_start = now;
    steps_at_action_start = step_counter;
}

int LearningSearch::epsilon_greedy_policy() {
    if (real_dist(rng) < EPSILON)
        // choose a random action
        return int_dist(rng);
    // choose the action with the highest weight
    return distance(weights.begin(), max_element(weights.begin(), weights.end()));
}

int LearningSearch::proportional_policy() {
    discrete_distribution<> dist(weights.begin(), weights.end());
    return dist(rng);
}

void LearningSearch::terminate_learning() {
    update_routine(); // reward the last routine
    ofstream weights_file("weights.txt");
    for (double w: weights)
        weights_file << w << " ";
    weights_file.close();
}

void LearningSearch::process_state(const SearchNode &node, const GlobalState &state,
    const GlobalOperator *op, const GlobalState &succ_state, bool is_preferred) {
    
    SearchNode succ_node = search_space.get_node(succ_state);

    if (succ_node.is_dead_end())
        return;

    if (succ_node.is_new()) {
        for (Heuristic *h: heuristics) {
            h->notify_state_transition(state, *op, succ_state);
        }
    }

    if (succ_node.is_new()) {
        // As in eager_search.cc, succ_node.get_g() isn't available yet
        int succ_g = node.get_g() + get_adjusted_cost(*op);

        // no preferred operators for now
        EvaluationContext eval_context(succ_state, succ_g, is_preferred, &statistics);
        statistics.inc_evaluated_states();

        if (open_list->is_dead_end(eval_context)) {
            succ_node.mark_as_dead_end();
            statistics.inc_dead_ends();
            return;
        }
        succ_node.open(node, op);

        // At the moment we only support a single heuristic
        int h = eval_context.get_heuristic_value(heuristics[0]);
        if (best_h > h) {
            best_h = h;
            if (all_time_best_h > h) {
                all_time_best_h = h;
                expansions_without_progress = 0;
            }
        }
        open_list->insert(eval_context, succ_state.get_id());
    } else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(*op)) {
        // A new cheapest path to an open or closed states
        if (reopen_closed_nodes) {
            if (succ_node.is_closed()) {
                statistics.inc_reopened();
            }
            succ_node.reopen(node, op);

            EvaluationContext eval_context(succ_state, succ_node.get_g(),
                is_preferred, &statistics);
                open_list->insert(eval_context, succ_state.get_id());
        } else {
            // As in eager_search.cc, if nodes aren't reopened we only change
            // the parent node.
            succ_node.update_parent(node, op);
        }
    }
}

// void LearningSearch::start_f_value_statistics(EvaluationContext &eval_context) {
//     if (f_evaluator) {
//         int f_value = eval_context.get_heuristic_value(f_evaluator);
//         statistics.report_f_value_progress(f_value);
//     }
// }

// void LearningSearch::update_f_value_statistics(const SearchNode &node) {
//     if (f_evaluator) {
//         EvaluationContext eval_context(node.get_state(), node.get_g(), false, &statistics);
//         int f_value = eval_context.get_heuristic_value(f_evaluator);
//         statistics.report_f_value_progress(f_value);
//     }
// }

shared_ptr<RAOpenListFactory> LearningSearch::create_simple_ra_open_list_factory(
    ScalarEvaluator *eval, bool pref_only) {

    Options state_list_opts;
    state_list_opts.set("eval", eval);
    state_list_opts.set("pref_only", pref_only);
    state_list_opts.set("epsilon", 0.2);
    
    return make_shared<SimpleRandomAccessOpenListFactory>(state_list_opts);
}

shared_ptr<RAOpenListFactory> LearningSearch::create_ra_alternation_open_list_factory(
    const vector<shared_ptr<RAOpenListFactory>> &subfactories, int boost) {
    
    Options options;
    options.set("sublists", subfactories);
    options.set("boost", boost);
    return make_shared<RAAlternationOpenListFactory>(options);
}


shared_ptr<RAOpenListFactory> LearningSearch::create_ra_open_list_factory (
    const Options &options) {
    
    const vector<ScalarEvaluator *> &evals = options.get_list<ScalarEvaluator*>("evals");
    const vector<Heuristic *> &preferred_heuristics = options.get_list<Heuristic*>("preferred");
    const int boost = options.get<int>("boost");

    if (evals.size() == 1 && preferred_heuristics.empty()) {
        return create_simple_ra_open_list_factory(evals[0], false);
    } else {
        vector<shared_ptr<RAOpenListFactory>> subfactories;
        for (ScalarEvaluator *evaluator: evals) {
            subfactories.push_back(
                create_simple_ra_open_list_factory(evaluator, false));
            if (!preferred_heuristics.empty()) {
                subfactories.push_back(
                    create_simple_ra_open_list_factory(evaluator, true));
            }
        }
        return create_ra_alternation_open_list_factory(subfactories, boost);
    }
}

void add_pruning_option(OptionParser &parser) {
    parser.add_option<shared_ptr<PruningMethod>>(
        "pruning",
        "Pruning methods can prune or reorder the set of applicable operators in "
        "each state and thereby influence the number and order of successor states "
        "that are considered.",
        "null()");
}

static SearchEngine *_parse(OptionParser &parser) {
    parser.document_synopsis("Learning search", "");
    parser.document_note(
        "Open list",
        "As in eager greedy best first search.");
    parser.document_note(
        "Closed nodes",
        "As in eager greedy best first search.");

    //parser.add_option<shared_ptr<RAOpenListFactory>>("ra_open", "random access open list");
    parser.add_list_option<ScalarEvaluator *>("evals", "scalar evaluators");
    parser.add_list_option<Heuristic *>(
        "preferred",
        "use preferred operators of these heuristics", "[]");
    parser.add_option<int>(
        "boost",
        "boost value for preferred operator open lists", "0");
    parser.add_option<double>("learning_rate", "the learning rate for RL", "0.001");

    add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);

    Options opts = parser.parse();
    opts.verify_list_non_empty<ScalarEvaluator *>("evals");

    LearningSearch *engine = nullptr;
    if (!parser.dry_run()) {
        opts.set("ra_open", LearningSearch::create_ra_open_list_factory(opts));
        opts.set("reopen_closed", false);
        opts.set("mpd", false);
        ScalarEvaluator *evaluator = nullptr;
        opts.set("f_eval", evaluator);
        engine = new LearningSearch(opts);
    }
    return engine;
}

static Plugin<SearchEngine> _plugin("learning", _parse);
}
