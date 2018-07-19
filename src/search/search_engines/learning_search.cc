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
      ref_time(opts.get<int>("t") / 2),
      n_states(compute_n_states()),
      weights_path(opts.get<string>("weights")),
      rng(system_clock::now().time_since_epoch().count()),
      //rng(0),
      trace("trace.txt"),
      learning_log("rl-log.txt"),
      action_count(actions.size(), 0),
      real_dist(0.0, 1.0),
      int_dist(0, actions.size()-1) {
    
    Options state_list_opts;
    const vector<ScalarEvaluator*> &evals =
            opts.get_list<ScalarEvaluator *>("evals");
    
    weights.reserve(n_states);
    auto initial_weights = vector<double>(actions.size(), INITIAL_WEIGHT);
    for (unsigned i = 0; i < n_states; ++i)
        weights.push_back(initial_weights);

    search_start = steady_clock::now();
    
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

    ifstream weight_file(weights_path);
    if(weight_file) {
        //weight_file >> avg_reward;
        for (vector<double> &row: weights)
            for (double &w: row)
                weight_file >> w;
        weight_file.close();
    }
    for (const vector<double> &row: weights) {
        auto p = get_probabilities(row);
        initial_distribution.push_back(
            discrete_distribution<>(p.begin(), p.end()));
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
        initial_h = all_time_best_h = best_h = previous_best_h = h;
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

    vector<const GlobalOperator*> applicable_ops;
    algorithms::OrderedSet<const GlobalOperator *> preferred_ops;
    StateID state_id = StateID::no_state;

    return expand(randomized, state_id, applicable_ops, preferred_ops);
}

SearchStatus LearningSearch::expand(bool randomized, StateID &state_id, vector<const GlobalOperator*> &applicable_ops,
    algorithms::OrderedSet<const GlobalOperator *> &preferred_ops) {
    
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

    state_id = state.get_id();
    g_successor_generator->generate_applicable_ops(state, applicable_ops);

    // Like in EagerSearch, this evaluates the expanded state (again) to get preferred ops.
    EvaluationContext eval_context(state, node.get_g(), false, &statistics, true);
    preferred_ops = collect_preferred_operators(eval_context, preferred_operator_heuristics);

    ++expansions_without_progress;
    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        bool is_preferred = preferred_ops.contains(op);
        process_state(node, state, op, succ_state, is_preferred);
    }

    return IN_PROGRESS;
}

SearchStatus LearningSearch::rollout_step() {

    vector<const GlobalOperator*> applicable_ops;
    algorithms::OrderedSet<const GlobalOperator *> preferred_operators;
    StateID state_id = StateID::no_state;

    SearchStatus status = expand(false, state_id, applicable_ops, preferred_operators);
    if (status != IN_PROGRESS)
        return status;

    if (applicable_ops.size() == 0 || expansions_without_progress < STALL_SIZE)
        return IN_PROGRESS;
    
    // Perform a stochastic rollout from expanded state
    GlobalState rollout_state = state_registry.lookup_state(state_id);
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
        bool is_preferred = preferred_operators.contains(op);
        EvaluationContext eval_context = process_state(node, rollout_state, op, succ_state, is_preferred, true);
        // Get the preferred operators for the next iteration.
        preferred_operators =
             collect_preferred_operators(eval_context, preferred_operator_heuristics);

        if(expansions_without_progress == 0)
            break;
        rollout_state = succ_state;
    }

    learning_log << endl;
    return IN_PROGRESS;
}

SearchStatus LearningSearch::preferred_rollout_step() {

    vector<const GlobalOperator*> applicable_ops;
    algorithms::OrderedSet<const GlobalOperator *> preferred_operators;
    StateID state_id = StateID::no_state;

    SearchStatus status = expand(false, state_id, applicable_ops, preferred_operators);
    if (status != IN_PROGRESS)
        return status;

    GlobalState state = state_registry.lookup_state(state_id);

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
        EvaluationContext eval_context = process_state(node, rollout_state, *it, succ_state, true, true);
        if (expansions_without_progress == 0)
            break;
        
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

SearchStatus LearningSearch::depth_first_step() {

    StateID state_id = StateID::no_state;
    if (dfs_stack.empty()) {
        pair<SearchNode, bool> n = fetch_next_node(false);
        if (!n.second) {
            return FAILED;
        }
        state_id = n.first.get_state().get_id();
    } else {
        state_id = dfs_stack.top();
        dfs_stack.pop();
    }

    GlobalState state = state_registry.lookup_state(state_id);
    SearchNode node = search_space.get_node(state);

    if (check_goal_and_set_plan(state)) {
        terminate_learning();
        return SOLVED;
    }

    vector<const GlobalOperator*> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);

    // Like in EagerSearch, this evaluates the expanded state (again) to get preferred ops.
    EvaluationContext eval_context(state, node.get_g(), false, &statistics, true);
    algorithms::OrderedSet<const GlobalOperator *> preferred_ops =
        collect_preferred_operators(eval_context, preferred_operator_heuristics);

    ++expansions_without_progress;
    vector<pair<StateID,int>> children;
    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        bool is_preferred = preferred_ops.contains(op);
        bool is_new = search_space.get_node(succ_state).is_new();
        EvaluationContext eval_context = process_state(node, state, op, succ_state, is_preferred);
        if (is_new) {
            int h = eval_context.get_heuristic_value(heuristics[0]);
            children.push_back(pair<StateID,int>(succ_state.get_id(), h));
        }
    }
    shuffle(children.begin(), children.end(), rng);
    sort(children.begin(), children.end(),
        [](const pair<StateID,int> &a, const pair<StateID,int> &b) {return a.second > b.second;});
    for (pair<StateID,int> c: children) {
        dfs_stack.push(c.first);
    }

    return IN_PROGRESS;
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
    // Clear the DFS stack
    while(!dfs_stack.empty())
        dfs_stack.pop();
    
    int reward = 0;
    if (step_counter > 0) {
        action_count[current_action_id] += 1;
        for (auto f: state)
            trace << f << " ";
        trace << current_action_id << endl;

        reward = previous_best_h - best_h;
        previous_best_h = best_h;

        // gradient_update(current_action_id, reward);
        
        rewards.push_back(reward);
    }

    state = get_state_features();
    current_action_id = initial_policy();

    // Dev logging
    steady_clock::time_point now = steady_clock::now();
    if (step_counter > 0) {
        cout << "Reward: " << reward;
        cout << ", Duration: "
            << duration_cast<milliseconds>(now - action_start).count();
        cout << ", Steps: " << step_counter - steps_at_action_start << endl;
    }
    cout << "Weights: " << setprecision(3);
    for(double d: weights[get_state_id()])
        cout << d << " ";
    cout << ", Choice: " << current_action_id << ", ";
    cout << "ENP: " << expansions_without_progress << ", ";

    action_start = now;
    steps_at_action_start = step_counter;
}

vector<unsigned> LearningSearch::get_state_features() {
    vector<unsigned> result;
    result.push_back(best_h > initial_h /2);
    result.push_back(
        duration_cast<milliseconds>(steady_clock::now()-search_start).count() > ref_time);
    return result;
}

unsigned LearningSearch::get_state_id() {
    auto features = get_state_features();
    unsigned result = 0;
    unsigned weight = 1;
    for (unsigned i = features.size()-1; i > 0; --i) {
        result += weight * features[i];
        weight *= STATE_SPACE[i];
    }
    return result;
}

// void LearningSearch::gradient_update(const int action_id, const int reward) {
//     double relative_reward = reward - avg_reward;
//     vector<double> probabilities = get_probabilities();
//     for (unsigned i = 0; i < weights.size(); ++i) {
//         if (i == (unsigned)action_id) {
//             weights[i] = weights[i] + learning_rate * relative_reward * (1 - probabilities[i]);
//         } else {
//             weights[i] = weights[i] - learning_rate * relative_reward * probabilities[i];
//         }
//     }
// }

int LearningSearch::uniform_policy() {
    return int_dist(rng);
}

int LearningSearch::proportional_policy() {
    unsigned state = get_state_id();
    discrete_distribution<> dist(weights[state].begin(), weights[state].end());
    return dist(rng);
}

int LearningSearch::softmax_policy() {
    unsigned state = get_state_id();
    vector<double> probabilities = get_probabilities(weights[state]);
    discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    return dist(rng);
}

int LearningSearch::initial_policy() {
    unsigned state = get_state_id();
    return initial_distribution[state](rng);
}

vector<double> LearningSearch::get_probabilities(const vector<double> &weights) {
    // TODO: Fine for weights up to 10, but consider a stable implementation. 
    vector<double> probabilities;
    probabilities.reserve(weights.size());
    double sum = 0.0;
    for (double w: weights) {
        double p = exp(w);
        probabilities.push_back(p);
        sum += p;
    }
    for (double &p: probabilities)
        p /= sum;
    return probabilities;
}

int LearningSearch::epsilon_greedy_policy() {
    if (real_dist(rng) < EPSILON)
        // choose a random action
        return uniform_policy();
    // choose the action with the highest weight
    unsigned state = get_state_id();
    return distance(weights[state].begin(),
        max_element(weights[state].begin(), weights[state].end()));
}

void LearningSearch::terminate_learning() {
    update_routine(); // reward the last routine
    // double avg_reward = 0.0;
    // for (int r: rewards)
    //     avg_reward += r;
    // if (rewards.size() > 1)
    //     avg_reward /= rewards.size();
    ofstream episode_file("episode.txt");
    // weights_file << avg_reward << endl;
    // for (double w: weights)
    //     weights_file << w << " ";
    // weights_file << endl;
    for (int c: action_count)
        episode_file << c << " ";
    episode_file.close();
}

EvaluationContext LearningSearch::process_state(const SearchNode &node, const GlobalState &state,
    const GlobalOperator *op, const GlobalState &succ_state, bool is_preferred, bool calculate_preferred) {
    
    SearchNode succ_node = search_space.get_node(succ_state);
    // As in eager_search.cc, succ_node.get_g() isn't available yet
    int succ_g = node.get_g() + get_adjusted_cost(*op);
    // no preferred operators for now
    EvaluationContext eval_context(succ_state, succ_g, is_preferred, &statistics, calculate_preferred);

    if (succ_node.is_dead_end())
        return eval_context;

    if (succ_node.is_new()) {
        for (Heuristic *h: heuristics) {
            h->notify_state_transition(state, *op, succ_state);
        }
    }

    if (succ_node.is_new()) {
    
        statistics.inc_evaluated_states();

        if (open_list->is_dead_end(eval_context)) {
            succ_node.mark_as_dead_end();
            statistics.inc_dead_ends();
            return eval_context;
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
        return eval_context;
    } else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(*op)) {
        // A new cheapest path to an open or closed states
        EvaluationContext eval_context(succ_state, succ_node.get_g(), // why with the old g?
                is_preferred, &statistics, calculate_preferred);
        if (reopen_closed_nodes) {
            if (succ_node.is_closed()) {
                statistics.inc_reopened();
            }
            succ_node.reopen(node, op);
            open_list->insert(eval_context, succ_state.get_id());
        } else {
            // As in eager_search.cc, if nodes aren't reopened we only change
            // the parent node.
            succ_node.update_parent(node, op);
        }
        return eval_context;
    }
    return eval_context;
}

unsigned LearningSearch::compute_n_states() {
    unsigned result = 1;
    for (unsigned u: STATE_SPACE)
        result *= u;
    return result;
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
    parser.add_option<string>("weights", "path to the weights file", "weights.txt");
    parser.add_option<int>("t", "time allocated for the search [ms]", "1000");

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
