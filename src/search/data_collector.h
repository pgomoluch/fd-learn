#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <iostream>

#include "search_engine.h"
#include "global_state.h"

class DataCollector
{
public:
    static void test();
    static void record_goal_path(SearchEngine *engine);
private:
    static void record_state(std::ostream &out, const GlobalState &state);
    static void record_data(std::ostream &out, const GlobalState &state, const int plan_length);
};

#endif
