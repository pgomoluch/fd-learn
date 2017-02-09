#ifndef HEURISTICS_FF_HEURISTIC_F_H
#define HEURISTICS_FF_HEURISTIC_F_H

#include "ff_heuristic.h"

namespace ff_heuristic {

class FFHeuristicF : public FFHeuristic {

public:
    FFHeuristicF(const options::Options &options);
    std::vector<double> get_features() { return features; }

protected:
    virtual int compute_heuristic(const GlobalState &global_state) override;

private:
    std::vector<double> features;
    //std::set<Proposition> delete_effects;
};

}

#endif
