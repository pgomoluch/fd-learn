#ifndef SEARCH_ENGINES_LEARNING_SEARCH_H
#define SEARCH_ENGINES_LEARNING_SEARCH_H

#include "../search_engine.h"

#include "../open_lists/open_list.h"
#include "../open_lists/ra_alternation_open_list.h"
#include "../open_lists/simple_random_access_open_list.h"

#include <chrono>
#include <deque>
#include <random>
#include <stack>

#include <fstream>

namespace options {
class Options;
}

namespace parametrized_search {

using RandomAccessStateOpenList = RandomAccessOpenList<StateOpenListEntry>;
using SimpleRandomAccessStateOpenList = SimpleRandomAccessOpenList<StateOpenListEntry>;

class ParametrizedSearch : public SearchEngine {
    const bool reopen_closed_nodes;
    
    double EPSILON = 0.5;
    unsigned ROLLOUT_LENGTH = 20;
    double P_LOCAL = 0.0;
    unsigned LOCAL_EXP_LIMIT = 100;

    const unsigned STALL_SIZE = 5;
    const std::vector<unsigned> STATE_SPACE = {2,2};

    RandomAccessStateOpenList *open_list;
    std::shared_ptr<RAOpenListFactory> open_list_factory;
    std::unique_ptr<RandomAccessStateOpenList> global_open_list;
    std::unique_ptr<RandomAccessStateOpenList> local_open_list;
    std::stack<StateOpenListEntry> dfs_stack;
    ScalarEvaluator *f_evaluator;
    std::vector<Heuristic*> heuristics;
    std::vector<Heuristic*> preferred_operator_heuristics;
    unsigned exp_since_local_restart = 0;
    unsigned local_exp = 0;
    unsigned steps_at_action_start = 0;
    unsigned expansions_without_progress = 0;
    int initial_h = -1;
    int best_h = -1;
    int previous_best_h = -1;
    int all_time_best_h = -1;
    const unsigned ref_time; // milliseconds
    const unsigned n_states;
    std::string params_path;
    std::chrono::steady_clock::time_point search_start; 
    double avg_reward = 0.0;
    std::vector<int> rewards;
    std::vector<unsigned> state;
    std::mt19937 rng;
    std::ofstream trace;
    std::ofstream learning_log;

    //void start_f_value_statistics(EvaluationContext &eval_context);
    //void update_f_value_statistics(const SearchNode &node);
    std::vector<unsigned> get_state_features();
    unsigned get_state_id();
    unsigned compute_n_states();
    
    void merge_local_list();
    void restart_local_list();
    
    // Gets one state from the queue and expands it. Sets state id, applicable and preferred operators,
    // which can be useful e.g. to perform a random walk from the expanded state.
    SearchStatus expand(bool randomized, StateID &state_id, std::vector<const GlobalOperator*> &applicable_ops,
        algorithms::OrderedSet<const GlobalOperator *> &preferred_ops);
    
    std::pair<SearchNode, bool> fetch_next_node(bool randomized);
    StateID get_best_state();
    StateID get_random_state();
    EvaluationContext process_state(const SearchNode &node, const GlobalState &state,
        const GlobalOperator *op, const GlobalState &succ_state,
        bool is_preferred = false, bool calculate_preferred = false);

    // High-level actions
    typedef SearchStatus (ParametrizedSearch::*Action)();

    SearchStatus random_walk(StateID &state_id, algorithms::OrderedSet<const GlobalOperator *> &preferred_operators);

    std::vector<Action> actions = {};
    //std::vector<std::vector<double>> weights;
    int current_action_id = 0;
    std::vector<int> action_count;
    std::uniform_real_distribution<> real_dist;
    std::chrono::steady_clock::time_point action_start;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit ParametrizedSearch(const options::Options &opts);

    virtual void print_statistics() const override;

    static std::shared_ptr<RAOpenListFactory> create_ra_open_list_factory(const Options &options);
    static std::shared_ptr<RAOpenListFactory> create_simple_ra_open_list_factory(
        ScalarEvaluator *eval, bool pref_only);
    static std::shared_ptr<RAOpenListFactory> create_ra_alternation_open_list_factory(
        const std::vector<std::shared_ptr<RAOpenListFactory>> &subfactories, int boost);
};

}

#endif
