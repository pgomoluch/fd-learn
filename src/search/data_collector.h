#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <iostream>

#include "search_engine.h"
#include "global_state.h"
#include "heuristics/ff_heuristic.h"

class DataCollector
{
public:
    DataCollector();
    static void test();
    void record_goal_path(SearchEngine *engine);
private:
    ff_heuristic::FFHeuristic ffh;
    static void record_state(std::ostream &out, const GlobalState &state);
    void record_data(std::ostream &fs, std::ostream &ls, const GlobalState &state, const int plan_cost, const int plan_length);
};

#endif
