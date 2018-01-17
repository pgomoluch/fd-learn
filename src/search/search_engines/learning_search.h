#ifndef SEARCH_ENGINES_LEARNING_SEARCH_H
#define SEARCH_ENGINES_LEARNING_SEARCH_H

#include "../search_engine.h"

#include "../open_lists/open_list.h"
#include "../open_lists/random_access_open_list.h"

#include <random>

namespace options {
class Options;
}

namespace learning_search {

using RandomAccessStateOpenList = RandomAccessOpenList<StateOpenListEntry>;

class LearningSearch : public SearchEngine {
    const bool reopen_closed_nodes;
    const unsigned STEP_SIZE = 100;
    const int INITIAL_WEIGHT = 10;
    std::unique_ptr<RandomAccessStateOpenList> open_list;
    ScalarEvaluator *f_evaluator;
    std::vector<Heuristic*> heuristics;
    unsigned step_counter = 0;
    std::mt19937 rng;

    std::pair<SearchNode, bool> fetch_next_node();
    //void start_f_value_statistics(EvaluationContext &eval_context);
    //void update_f_value_statistics(const SearchNode &node);

    // High-level actions
    typedef StateID (LearningSearch::*Action)();

    StateID get_best_state();
    StateID get_randomized_state();

    std::vector<Action> actions = {
        &LearningSearch::get_best_state,
        &LearningSearch::get_randomized_state};
    std::vector<int> weights = {INITIAL_WEIGHT, INITIAL_WEIGHT};
    int current_action_id = 0;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit LearningSearch(const options::Options &opts);

    virtual void print_statistics() const override;
};

}

#endif
