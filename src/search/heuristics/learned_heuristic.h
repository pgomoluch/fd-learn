#ifndef HEURISTICS_LEARNED_HEURISTIC_H
#define HEURISTICS_LEARNED_HEURISTIC_H

#include <vector>

#include "../heuristic.h"
#include "../state_encoder.h"

namespace learned_heuristic {

class LearnedHeuristic : public Heuristic {
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    LearnedHeuristic(const options::Options &options);
    ~LearnedHeuristic();
private:
    StateEncoder state_encoder;
    std::vector<double> model;
    double intercept;
};

}

#endif
