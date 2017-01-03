#ifndef HEURISTICS_LEARNED_HEURISTIC_H
#define HEURISTICS_LEARNED_HEURISTIC_H

#include <vector>

#include "../heuristic.h"
#include "../data_collector.h"

namespace learned_heuristic {

class LearnedHeuristic : public Heuristic {
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    LearnedHeuristic(const options::Options &options);
    ~LearnedHeuristic();
private:
    DataCollector data_collector;
    std::vector<double> model;
};

}

#endif
