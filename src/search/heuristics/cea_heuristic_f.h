#ifndef HEURISTICS_CEA_HEURISTIC_F_H
#define HEURISTICS_CEA_HEURISTIC_F_H

#include "../domain_transition_graph.h"
#include "../heuristic.h"
#include "../priority_queue.h"

#include <map>
#include <vector>

class State;

namespace cea_heuristic_f {
struct LocalProblem;
struct LocalProblemNode;
struct LocalTransition;

class ContextEnhancedAdditiveHeuristicF : public Heuristic {
    std::vector<DomainTransitionGraph *> transition_graphs;
    std::vector<LocalProblem *> local_problems;
    std::vector<std::vector<LocalProblem *>> local_problem_index;
    LocalProblem *goal_problem;
    LocalProblemNode *goal_node;
    int min_action_cost;

    AdaptiveQueue<LocalProblemNode *> node_queue;

    std::vector<double> features;
    std::vector<double> schema_count;
    std::vector<std::vector<bool>> pairwise_features;    
    std::map<std::string, int> schema_map;
    int max_graph_depth;

    LocalProblem *get_local_problem(int var_no, int value);
    LocalProblem *build_problem_for_variable(int var_no) const;
    LocalProblem *build_problem_for_goal() const;

    int get_priority(LocalProblemNode *node) const;
    void initialize_heap();
    void add_to_heap(LocalProblemNode *node);

    bool is_local_problem_set_up(const LocalProblem *problem) const;
    void set_up_local_problem(LocalProblem *problem, int base_priority,
                              int start_value, const State &state);

    void try_to_fire_transition(LocalTransition *trans);
    void expand_node(LocalProblemNode *node);
    void expand_transition(LocalTransition *trans, const State &state);

    int compute_costs(const State &state);
    void mark_helpful_transitions(
        LocalProblem *problem, LocalProblemNode *node, const State &state);
    // Clears "first_on_path" of visited nodes as a side effect to avoid
    // recursing to the same node again.
    
    string get_schema_name(const OperatorProxy &op);
    void compute_features(LocalProblem *problem, LocalProblemNode *node,
        const State &state);
    void set_features_from_graph(LocalProblem *problem, LocalProblemNode *node,
        const State &state, std::vector<OperatorProxy> supported_ops, int depth);
    
protected:
    virtual int compute_heuristic(const GlobalState &state);
public:
    explicit ContextEnhancedAdditiveHeuristicF(const options::Options &opts);
    ~ContextEnhancedAdditiveHeuristicF();
    virtual bool dead_ends_are_reliable() const;
    std::vector<double> get_features() { return features; }
    std::vector<double> get_dd_features();
};
}

#endif
