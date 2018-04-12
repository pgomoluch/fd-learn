#ifndef SEARCH_ENGINES_LEARNING_SEARCH_H
#define SEARCH_ENGINES_LEARNING_SEARCH_H

#include "../search_engine.h"

#include "../open_lists/open_list.h"
#include "../open_lists/simple_random_access_open_list.h"

#include <random>
#include <deque>

#include <fstream>

namespace options {
class Options;
}

namespace learning_search {

using RandomAccessStateOpenList = RandomAccessOpenList<StateOpenListEntry>;
using SimpleRandomAccessStateOpenList = SimpleRandomAccessOpenList<StateOpenListEntry>;

class LearningSearch : public SearchEngine {
    const bool reopen_closed_nodes;
    const unsigned STEP_SIZE = 100;
    const int INITIAL_WEIGHT = 40;
    const unsigned REWARD_WINDOW = 2;
    const int UNIT_REWARD = 4;
    const unsigned ROLLOUT_LENGTH = 20;
    const unsigned STALL_SIZE = 1000;
    std::unique_ptr<RandomAccessStateOpenList> open_list;
    ScalarEvaluator *f_evaluator;
    std::vector<Heuristic*> heuristics;
    std::vector<Heuristic*> preferred_operator_heuristics;
    unsigned step_counter = 0;
    unsigned expansions_without_progress = 0;
    int best_h = -1;
    int previous_best_h = -1;
    int all_time_best_h = -1;
    std::mt19937 rng;
    std::ofstream learning_log;

    //void start_f_value_statistics(EvaluationContext &eval_context);
    //void update_f_value_statistics(const SearchNode &node);
    void update_routine();
    
    SearchStatus simple_step(bool randomized);
    std::pair<SearchNode, bool> fetch_next_node(bool randomized);
    StateID get_best_state();
    StateID get_randomized_state();
    void process_state(const SearchNode &node, const GlobalState &state,
        const GlobalOperator *op, const GlobalState &succ_state,
        bool is_preferred = false);

    // High-level actions
    typedef SearchStatus (LearningSearch::*Action)();

    SearchStatus greedy_step();
    SearchStatus epsilon_greedy_step();
    SearchStatus rollout_step();
    SearchStatus preferred_rollout_step();

    std::vector<Action> actions = {
        &LearningSearch::greedy_step,
        &LearningSearch::epsilon_greedy_step,
        &LearningSearch::rollout_step,
        &LearningSearch::preferred_rollout_step};
    std::vector<int> weights = {INITIAL_WEIGHT, INITIAL_WEIGHT, INITIAL_WEIGHT, INITIAL_WEIGHT};
    std::deque<int> past_actions;
    int current_action_id = 0;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit LearningSearch(const options::Options &opts);

    virtual void print_statistics() const override;

    static std::shared_ptr<RAOpenListFactory> create_ra_open_list_factory(const Options &options);
    static std::shared_ptr<RAOpenListFactory> create_simple_ra_open_list_factory(
        ScalarEvaluator *eval, bool pref_only);
    static std::shared_ptr<RAOpenListFactory> create_ra_alternation_open_list_factory(
        const std::vector<std::shared_ptr<RAOpenListFactory>> &subfactories, int boost);
};

}

#endif
