#include "features.h"

#include "global_operator.h"
#include "globals.h"

namespace features
{
    int distance(const GlobalState &state)
    {
        int distance = 0;
        for (size_t i = 0; i < g_goal.size(); ++i)
            if(state.get_values()[g_goal[i].first] != g_goal[i].second)
                ++distance;
        
        return distance;
    }
    
    int applicable_operator_count(const GlobalState &state)
    {
        int count = 0;
        for(auto it = g_operators.begin(); it != g_operators.end(); ++it)
        {
            if(it->is_applicable(state))
                ++count;
        }
        return count;
    }
    
    int non_diverging_operator_count(const GlobalState &state)
    {
        int count = 0;
        for(auto it = g_operators.begin(); it != g_operators.end(); ++it)
        {
            if(it->is_applicable(state))
                if(!diverges_from_goal(*it, state))
                    count++;
        }
        return count;
    }
       
    bool diverges_from_goal(const GlobalOperator &op, const GlobalState &state)
    {
        for(auto it = op.get_effects().begin(); it != op.get_effects().end(); ++it)
        {
            if(it->does_fire(state) && defines_goal(it->var))
                if(conjunct_satisfied(it->var, state) && state[it->var] != it->val)
                    return true; 
        }
        return false;
    }
    
    bool defines_goal(int var)
    {
        for(size_t i = 0; i < g_goal.size(); ++i)
            if(g_goal[i].first == var)
                return true;
        return false;
    }
    
    bool conjunct_satisfied(int var, const GlobalState &state)
    {
         for(size_t i = 0; i < g_goal.size(); ++i)
            if(g_goal[i].first == var)
                return state[i] == g_goal[i].second;
        return true;
    }
}
