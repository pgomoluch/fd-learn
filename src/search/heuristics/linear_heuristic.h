#ifndef HEURISTICS_LINEAR_HEURISTIC_H
#define HEURISTICS_LINEAR_HEURISTIC_H

#include <vector>

#include "../heuristic.h"
#include "../state_encoder.h"

namespace linear_heuristic {

class LinearHeuristic : public Heuristic {
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    LinearHeuristic(const options::Options &options);
    ~LinearHeuristic();
private:
    StateEncoder state_encoder;
    std::vector<double> model;
    double intercept;
};

}

#endif
