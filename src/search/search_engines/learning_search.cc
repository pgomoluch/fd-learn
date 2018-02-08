#include "learning_search.h"

#include "search_common.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"
#include "../successor_generator.h"

#include "../open_lists/open_list_factory.h"
#include "../utils/memory.h"

#include <random>

using namespace std;

namespace learning_search {

LearningSearch::LearningSearch(const Options &opts)
    : SearchEngine(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      //open_list(opts.get<shared_ptr<OpenListFactory>>("open")->
      //  create_state_open_list()),
      f_evaluator(opts.get<ScalarEvaluator *>("f_eval", nullptr)),
      rng(0) {
    
    Options state_list_opts;
    const vector<ScalarEvaluator*> &evals =
            opts.get_list<ScalarEvaluator *>("evals");
    // At the moment we only support a single evaluator
    state_list_opts.set("eval", evals[0]);
    state_list_opts.set("pref_only", false);
    state_list_opts.set("epsilon", 0.2);
    open_list =
        utils::make_unique_ptr<RandomAccessStateOpenList>(state_list_opts);
}

void LearningSearch::initialize() {
    cout << "Conducting learning search..." << endl;
    assert(open_list);

    set<Heuristic*> hset;
    open_list->get_involved_heuristics(hset);

    heuristics.assign(hset.begin(), hset.end());
    assert(!heuristics.empty());

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

        open_list->insert(eval_context, initial_state.get_id());
    }
}

void LearningSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

SearchStatus LearningSearch::step() {
    if (step_counter % STEP_SIZE == 0)
        update_routine();
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
    if (check_goal_and_set_plan(state))
        return SOLVED;

    vector<const GlobalOperator*> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);

    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        process_state(node, state, op, succ_state);
    }

    ++step_counter;
    return IN_PROGRESS;
}

SearchStatus LearningSearch::rollout_step() {

    pair<SearchNode, bool> n = fetch_next_node(false);
    if (!n.second) {
        return FAILED;
    }
    SearchNode node = n.first;

    GlobalState state = node.get_state();
    if (check_goal_and_set_plan(state))
        return SOLVED;
    
    vector<const GlobalOperator*> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);
    
    // Fully expand the processed state
    for (const GlobalOperator *op: applicable_ops) {
        GlobalState succ_state = state_registry.get_successor_state(state, *op);
        statistics.inc_generated();
        process_state(node, state, op, succ_state);
    }

    if (applicable_ops.size() == 0)
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
        process_state(node, rollout_state, op, succ_state);
        rollout_state = succ_state;
    }

    ++step_counter;
    return IN_PROGRESS;
}

pair<SearchNode, bool> LearningSearch::fetch_next_node(bool randomized) {
    while (true) {
        if (open_list->empty()) {
            cout << "Completely explored state space -- no solution!" << endl;
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
    int reward = 0;
    if (step_counter > 0) {
        reward = open_list->get_reward();
        open_list->reset_reward();

        if (reward > 0) {
            int update = UNIT_REWARD;
            // reward the recent action
            weights[current_action_id] += update;
            // reward up to REWARD_WINDOW previous actions
            for (auto it = past_actions.begin(); it != past_actions.end(); ++it) {
                update /= 2;
                weights[*it] += update;
            }
        }
    }
    // remember up to REWARD_WINDOW previous actions
    past_actions.push_front(current_action_id);
    if (past_actions.size() > REWARD_WINDOW)
        past_actions.pop_back();
    
    // choose the next action
    discrete_distribution<> d(weights.begin(), weights.end());
    current_action_id = d(rng);
    
    // Dev logging
    cout << "Reward: " << reward << endl;
    cout << "Weights: ";
    for(int i: weights)
        cout << i << " ";
    cout << ", Choice: " << current_action_id << ", ";
}

void LearningSearch::process_state(const SearchNode &node, const GlobalState &state,
    const GlobalOperator *op, const GlobalState &succ_state) {
    
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
        EvaluationContext eval_context(succ_state, succ_g, false, &statistics);
        statistics.inc_evaluated_states();

        if (open_list->is_dead_end(eval_context)) {
            succ_node.mark_as_dead_end();
            statistics.inc_dead_ends();
            return;
        }
        succ_node.open(node, op);

        open_list->insert(eval_context, succ_state.get_id());
    } else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(*op)) {
        // A new cheapest path to an open or closed states
        if (reopen_closed_nodes) {
            if (succ_node.is_closed()) {
                statistics.inc_reopened();
            }
            succ_node.reopen(node, op);

            EvaluationContext eval_context(succ_state, succ_node.get_g(),
                false, &statistics);
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

    parser.add_list_option<ScalarEvaluator *>("evals", "scalar evaluators");
    parser.add_list_option<Heuristic *>(
        "preferred",
        "use preferred operators of these heuristics", "[]");
    parser.add_option<int>(
        "boost",
        "boost value for preferred operator open lists", "0");

    add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);

    Options opts = parser.parse();
    opts.verify_list_non_empty<ScalarEvaluator *>("evals");

    LearningSearch *engine = nullptr;
    if (!parser.dry_run()) {
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
