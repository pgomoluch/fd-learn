#ifndef OPEN_LISTS_RANDOM_ACCESS_OPEN_LIST_H
#define OPEN_LISTS_RANDOM_ACCESS_OPEN_LIST_H

/*
  Open list allowing for random access to stored elements, developed for
  use with the learning search engine (LearningSearch class). Starts as
  a copy of EpsilonGreedyOpenList, but is expected to diverge quickly.
*/

#include "open_list_factory.h"
#include "../option_parser_util.h"

class RandomAccessOpenListFactory : public OpenListFactory {
    Options options; // ?
public:
    explicit RandomAccessOpenListFactory(const Options &options);
    virtual ~RandomAccessOpenListFactory() override = default;

    virtual std::unique_ptr<StateOpenList> create_state_open_list() override;
    virtual std::unique_ptr<EdgeOpenList> create_edge_open_list() override;
};

/*
  Unlike other open list implementations, RandomAccessOpenList class is
  defined in the header so that its full type is accessible to the search
  engine. The reason is that its public interface exceeds the one defined
  by OpenList class template.
*/

template<class Entry>
class RandomAccessOpenList : public OpenList<Entry> {
    struct HeapNode {
        int id;
        int h;
        Entry entry;
        HeapNode(int id, int h, const Entry &entry)
            : id(id), h(h), entry(entry) {
        }

        bool operator>(const HeapNode &other) const {
            return std::make_pair(h, id) > std::make_pair(other.h, other.id);
        }
    };

    std::vector<HeapNode> heap;
    ScalarEvaluator *evaluator;

    double epsilon;
    int size;
    int next_id;
    int best_h = -1;
    int previous_best_h = -1;

protected:
    virtual void do_insertion(EvaluationContext &eval_context,
                              const Entry &entry) override;

public:
    explicit RandomAccessOpenList(const Options &opts);
    virtual ~RandomAccessOpenList() override = default;

    // Open list interface
    virtual Entry remove_min(std::vector<int> *key = nullptr) override;
    virtual bool is_dead_end(
        EvaluationContext &eval_context) const override;
    virtual bool is_reliable_dead_end(
        EvaluationContext &eval_context) const override;
    virtual void get_involved_heuristics(std::set<Heuristic *> &hset) override;
    virtual bool empty() const override;
    virtual void clear() override;

    // Remaining methods
    Entry remove_random(std::vector<int> *key = nullptr);
    Entry remove_epsilon(std::vector<int> *key = nullptr);
    void reset_reward();
    int get_reward();
};

#endif