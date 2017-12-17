#ifndef SEARCH_ENGINES_LEARNING_SEARCH_H
#define SEARCH_ENGINES_LEARNING_SEARCH_H

#include "../search_engine.h"

#include "../open_lists/open_list.h"

namespace options {
class Options;
}

namespace learning_search {
class LearningSearch : public SearchEngine {
    const bool reopen_closed_nodes;
    std::unique_ptr<StateOpenList> open_list;
    ScalarEvaluator *f_evaluator;
    std::vector<Heuristic*> heuristics;

    std::pair<SearchNode, bool> fetch_next_node();
    //void start_f_value_statistics(EvaluationContext &eval_context);
    //void update_f_value_statistics(const SearchNode &node);
protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit LearningSearch(const options::Options &opts);

    virtual void print_statistics() const override;
};

}

#endif
