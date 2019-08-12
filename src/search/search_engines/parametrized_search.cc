#include "parametrized_search.h"

#include "search_common.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"
#include "../successor_generator.h"

#include "../open_lists/ra_alternation_open_list.h"
#include "../utils/memory.h"

#include <cmath>
#include <iomanip>

using namespace std;
using namespace std::chrono;

namespace parametrized_search {

ParametrizedSearch::ParametrizedSearch(const Options &opts)
    : SearchEngine(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      open_list_factory(opts.get<shared_ptr<RAOpenListFactory>>("ra_open")),
      global_open_list(open_list_factory->create_state_open_list()),
      local_open_list(open_list_factory->create_state_open_list()),
      f_evaluator(opts.get<ScalarEvaluator *>("f_eval", nullptr)),
      preferred_operator_heuristics(opts.get_list<Heuristic *>("preferred")),
      ref_time(opts.get<int>("t") / 2),
      params_path(opts.get<string>("params")),
      rng(system_clock::now().time_since_epoch().count()),
      //rng(0),
      //learning_log("rl-log.txt"),
      real_dist(0.0, 1.0)
{
    
    Options state_list_opts;
    const vector<ScalarEvaluator*> &evals =
            opts.get_list<ScalarEvaluator *>("evals");

    search_start = steady_clock::now();
    
    open_list = global_open_list.get();
}

void ParametrizedSearch::initialize() {
    cout << "Conducting parametrized search...";
    assert(open_list);

    set<Heuristic*> hset;
    open_list->get_involved_heuristics(hset);

    hset.insert(preferred_operator_heuristics.begin(),
        preferred_operator_heuristics.end());

    heuristics.assign(hset.begin(), hset.end());
    assert(!heuristics.empty());

    if (neural_parametrized) {
        nn = unique_ptr<Network>(new Network(params_path.c_str(), true));
    } else {
        ifstream params_file(params_path);
        if(params_file) {
            params_file >> EPSILON;
            params_file >> STALL_SIZE;
            params_file >> N_ROLLOUTS;
            params_file >> ROLLOUT_LENGTH;

            unsigned cycle_length;
            double percentage_local;
            params_file >> cycle_length;
            params_file >> percentage_local;
            // cycle_length == GLOBAL_EXP_LIMIT + LOCAL_EXP_LIMIT
            GLOBAL_EXP_LIMIT = cycle_length * (1 - percentage_local);
            LOCAL_EXP_LIMIT = cycle_length - GLOBAL_EXP_LIMIT;

            params_file.close();
        }
        cout << "\nepsilon = " << EPSILON;
        cout << "\nrollout_length = " << ROLLOUT_LENGTH;
        cout << "\nglobal_exp_limit = " << GLOBAL_EXP_LIMIT;
        cout << "\nlocal_exp_limit = " << LOCAL_EXP_LIMIT;
        cout << endl;
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
        initial_h = best_h = h;
        open_list->insert(eval_context, initial_state.get_id());
    }

    if (neural_parametrized)
        update_search_parameters();

}

void ParametrizedSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

SearchStatus ParametrizedSearch::step() {

    if (open_list == local_open_list.get() && exp_since_switch >= LOCAL_EXP_LIMIT) {
        merge_local_list();
        open_list = global_open_list.get();
        exp_since_switch = 0;
        update_search_parameters();
    } else if (open_list == global_open_list.get() && exp_since_switch >= GLOBAL_EXP_LIMIT) {
        restart_local_list();
        open_list = local_open_list.get();
        exp_since_switch = 0;
    }

    vector<const GlobalOperator*> applicable_ops;
    algorithms::OrderedSet<const GlobalOperator *> preferred_operators;
    StateID state_id = StateID::no_state;

    bool expand_random = (real_dist(rng) < EPSILON);
    ++exp_since_switch;
    SearchStatus status = expand(expand_random, state_id, applicable_ops, preferred_operators);
    
    if (status == FAILED) {
        // When local search fails, try to get another state from the global queue
        if (global_open_list->empty())
            return FAILED;
        else {
            restart_local_list();
            return IN_PROGRESS;
        }
    }

    if (status != IN_PROGRESS)
        return status;

    if (applicable_ops.size() == 0 || expansions_without_progress < STALL_SIZE)
        return IN_PROGRESS;
    for (unsigned i = 0; i < N_ROLLOUTS; ++i) {
        //learning_log << i << ' ';
        random_walk(state_id, preferred_operators);
        if (expansions_without_progress == 0)
            break;
    }
    //learning_log << "\n";
    return IN_PROGRESS;
}

SearchStatus ParametrizedSearch::expand(bool randomized, StateID &state_id, vector<const GlobalOperator*> &applicable_ops,
    algorithms::OrderedSet<const GlobalOperator *> &preferred_ops) {
    
    pair<SearchNode, bool> n = fetch_next_node(randomized);
    if (!n.second) {
        return FAILED;
    }
    SearchNode node = n.first;

    GlobalState state = node.get_state();
    if (check_goal_and_set_plan(state)) {
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

SearchStatus ParametrizedSearch::random_walk(StateID &state_id, algorithms::OrderedSet<const GlobalOperator *> &preferred_operators) {
    
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
        //learning_log << "#";
        bool is_preferred = preferred_operators.contains(op);
        EvaluationContext eval_context = process_state(node, rollout_state, op, succ_state, is_preferred, true);
        // Get the preferred operators for the next iteration.
        preferred_operators =
             collect_preferred_operators(eval_context, preferred_operator_heuristics);

        if(expansions_without_progress == 0)
            break;
        rollout_state = succ_state;
    }

    //learning_log << endl;
    return IN_PROGRESS;
}

void ParametrizedSearch::restart_local_list() {
    open_list = global_open_list.get();
    StateID id = get_best_state();
    GlobalState state = state_registry.lookup_state(id);
    SearchNode node = search_space.get_node(state);

    local_open_list = open_list_factory->create_state_open_list();
    EvaluationContext eval_context(state, node.get_g(), true, &statistics);
    local_open_list->insert(eval_context, id);
}

void ParametrizedSearch::merge_local_list() {
    while (!local_open_list->empty()) {
        StateID id = local_open_list->remove_min();
        GlobalState state = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(state);
        // None of the local search states are preferred
        EvaluationContext eval_context(state, node.get_g(), false, &statistics);
        global_open_list->insert(eval_context, id);
    }
}

pair<SearchNode, bool> ParametrizedSearch::fetch_next_node(bool randomized) {
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
            id = get_random_state();
        else
            id = get_best_state();

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

StateID ParametrizedSearch::get_best_state() {
    return open_list->remove_min();
}

StateID ParametrizedSearch::get_random_state() {
    return open_list->remove_random();
}

vector<double> ParametrizedSearch::get_state_features() {
    vector<double> result;
    
    result.push_back((double) initial_h);
    result.push_back((double) best_h);
    result.push_back((double)
        duration_cast<seconds>(steady_clock::now()-search_start).count());
    result.push_back((double) expansions_without_progress);

    result.push_back((double)statistics.get_generated());
    result.push_back((double)statistics.get_evaluated_states()); // unlike generated, this excludes duplicates
    result.push_back((double)statistics.get_expanded());
    
    return result;
}

void ParametrizedSearch::update_search_parameters()
{
    auto features = get_state_features();
    //for (auto f: features)
    //    learning_log << f << " ";
    for (unsigned i = 0; i < features.size(); ++i)
        features[i] /= FEATURE_SCALES[i];
        
    Matrix result(6, 1);
    nn->evaluate(features, result);
    
    auto sigma = [](double x) { return 1.0 / (1.0 + exp(-x)); };
    auto nonneg = [](double x) { return x < 0.0 ? 0.0 : x; };

    EPSILON = sigma(result.at(0, 0));
    STALL_SIZE = nonneg(result.at(1, 0)) * STALL_SIZE_SCALE;
    N_ROLLOUTS = nonneg(result.at(2, 0)) * N_ROLLOUTS_SCALE;
    ROLLOUT_LENGTH = nonneg(result.at(3, 0)) * ROLLOUT_LENGTH_SCALE;
    
    double cycle_length = nonneg(result.at(4, 0)) * GLOBAL_LOCAL_CYCLE_SCALE;
    double percentage_local = sigma(result.at(5, 0));
    GLOBAL_EXP_LIMIT = cycle_length * (1 - percentage_local);
    LOCAL_EXP_LIMIT = cycle_length - GLOBAL_EXP_LIMIT;

    //for (auto f: features)
    //    learning_log << f << " ";
    //learning_log << endl;
    //learning_log << EPSILON << " " << STALL_SIZE << " " << N_ROLLOUTS << " "
    //    << ROLLOUT_LENGTH << " " << GLOBAL_EXP_LIMIT << " " << LOCAL_EXP_LIMIT << endl << endl; 
}

EvaluationContext ParametrizedSearch::process_state(const SearchNode &node, const GlobalState &state,
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
            expansions_without_progress = 0;
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

shared_ptr<RAOpenListFactory> ParametrizedSearch::create_simple_ra_open_list_factory(
    ScalarEvaluator *eval, bool pref_only) {

    Options state_list_opts;
    state_list_opts.set("eval", eval);
    state_list_opts.set("pref_only", pref_only);
    state_list_opts.set("epsilon", 0.2);
    
    return make_shared<SimpleRandomAccessOpenListFactory>(state_list_opts);
}

shared_ptr<RAOpenListFactory> ParametrizedSearch::create_ra_alternation_open_list_factory(
    const vector<shared_ptr<RAOpenListFactory>> &subfactories, int boost) {
    
    Options options;
    options.set("sublists", subfactories);
    options.set("boost", boost);
    return make_shared<RAAlternationOpenListFactory>(options);
}


shared_ptr<RAOpenListFactory> ParametrizedSearch::create_ra_open_list_factory (
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
    parser.document_synopsis("Parametrized search", "");
    parser.document_note(
        "Open list",
        "As in eager greedy best first search.");
    parser.document_note(
        "Closed nodes",
        "As in eager greedy best first search.");

    parser.add_list_option<ScalarEvaluator *>("evals", "scalar evaluators");
    parser.add_list_option<Heuristic *>(
        "preferred",
        "use preferred operators of these heuristics", "[]");
    parser.add_option<int>(
        "boost",
        "boost value for preferred operator open lists", "0");
    parser.add_option<string>("params", "path to the weights file", "params.txt");
    parser.add_option<int>("t", "time allocated for the search [ms]", "1000");

    add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);

    Options opts = parser.parse();
    opts.verify_list_non_empty<ScalarEvaluator *>("evals");

    ParametrizedSearch *engine = nullptr;
    if (!parser.dry_run()) {
        opts.set("ra_open", ParametrizedSearch::create_ra_open_list_factory(opts));
        opts.set("reopen_closed", false);
        opts.set("mpd", false);
        ScalarEvaluator *evaluator = nullptr;
        opts.set("f_eval", evaluator);
        engine = new ParametrizedSearch(opts);
    }
    return engine;
}

static Plugin<SearchEngine> _plugin("parametrized", _parse);
}
