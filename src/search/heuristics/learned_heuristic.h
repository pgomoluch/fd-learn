#ifndef HEURISTICS_LEARNED_HEURISTIC_H
#define HEURISTICS_LEARNED_HEURISTIC_H

#include "../heuristic.h"

namespace learned_heuristic {

class LearnedHeuristic : public Heuristic {
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    LearnedHeuristic(const options::Options &options);
    ~LearnedHeuristic();
};

}

#endif
