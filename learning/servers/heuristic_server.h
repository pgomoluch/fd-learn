#ifndef HEURISTIC_SERVER_H
#define HEURISTIC_SERVER_H

#include <vector>

class HeuristicServer
{
public:
    HeuristicServer(int n_features);
    int serve();
protected:
    virtual double evaluate(double state[]) = 0;
    const int n_features;
};

#endif
