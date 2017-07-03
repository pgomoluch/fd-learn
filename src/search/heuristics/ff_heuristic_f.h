#ifndef HEURISTICS_FF_HEURISTIC_F_H
#define HEURISTICS_FF_HEURISTIC_F_H

#include "ff_heuristic.h"

#include <map>

namespace ff_heuristic {

class FFHeuristicF : public FFHeuristic {

public:
    FFHeuristicF(const options::Options &options);
    std::vector<double> get_features() { return features; }
    std::vector<double> get_dd_features() { return dd_features; }

protected:
    void mark_preferred_operators_and_relaxed_plan_f(
        const State &state, Proposition *goal,
        std::vector<UnaryOperator*> supported_ops = std::vector<UnaryOperator*>());
    virtual int compute_heuristic(const GlobalState &global_state) override;

private:
    std::vector<double> features;
    std::vector<double> dd_features; // domain-dependent features
    std::vector<std::vector<bool>> pairwise_features;
    
    std::map<std::string, int> schema_map;
    
    static std::string get_schema_name(OperatorProxy op);
    //std::set<Proposition> delete_effects;
};

}

#endif
