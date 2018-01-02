#include "random_access_open_list.h"

#include "../utils/memory.h"

using namespace std;

template<class Entry>
class RandomAccessOpenList : public OpenList<Entry> {
    typedef deque<Entry> Bucket;

    map<int, Bucket> buckets;
    int size;

    ScalarEvaluator *evaluator;

protected:
    virtual void do_insertion(EvaluationContext &eval_context,
                              const Entry &entry) override;

public:
    explicit RandomAccessOpenList(const Options &opts);
    RandomAccessOpenList(ScalarEvaluator *eval,
                           bool preferred_only);
    virtual ~RandomAccessOpenList() override = default;

    virtual Entry remove_min(vector<int> *key = nullptr) override;
    virtual bool empty() const override;
    virtual void clear() override;
    virtual void get_involved_heuristics(set<Heuristic *> &hset) override;
    virtual bool is_dead_end(
        EvaluationContext &eval_context) const override;
    virtual bool is_reliable_dead_end(
        EvaluationContext &eval_context) const override;
};


template<class Entry>
RandomAccessOpenList<Entry>::RandomAccessOpenList(const Options &opts)
    : OpenList<Entry>(opts.get<bool>("pref_only")),
      size(0),
      evaluator(opts.get<ScalarEvaluator *>("eval")) {
}

template<class Entry>
RandomAccessOpenList<Entry>::RandomAccessOpenList(
    ScalarEvaluator *evaluator, bool preferred_only)
    : OpenList<Entry>(preferred_only),
      size(0),
      evaluator(evaluator) {
}

template<class Entry>
void RandomAccessOpenList<Entry>::do_insertion(
    EvaluationContext &eval_context, const Entry &entry) {
    int key = eval_context.get_heuristic_value(evaluator);
    buckets[key].push_back(entry);
    ++size;
}

template<class Entry>
Entry RandomAccessOpenList<Entry>::remove_min(vector<int> *key) {
    assert(size > 0);
    auto it = buckets.begin();
    assert(it != buckets.end());
    if (key) {
        assert(key->empty());
        key->push_back(it->first);
    }

    Bucket &bucket = it->second;
    assert(!bucket.empty());
    Entry result = bucket.front();
    bucket.pop_front();
    if (bucket.empty())
        buckets.erase(it);
    --size;
    return result;
}

template<class Entry>
bool RandomAccessOpenList<Entry>::empty() const {
    return size == 0;
}

template<class Entry>
void RandomAccessOpenList<Entry>::clear() {
    buckets.clear();
    size = 0;
}

template<class Entry>
void RandomAccessOpenList<Entry>::get_involved_heuristics(
    set<Heuristic *> &hset) {
    evaluator->get_involved_heuristics(hset);
}

template<class Entry>
bool RandomAccessOpenList<Entry>::is_dead_end(
    EvaluationContext &eval_context) const {
    return eval_context.is_heuristic_infinite(evaluator);
}

template<class Entry>
bool RandomAccessOpenList<Entry>::is_reliable_dead_end(
    EvaluationContext &eval_context) const {
    return is_dead_end(eval_context) && evaluator->dead_ends_are_reliable();
}

RandomAccessOpenListFactory::RandomAccessOpenListFactory(
    const Options &options)
    : options(options) {
}

unique_ptr<StateOpenList>
RandomAccessOpenListFactory::create_state_open_list() {
    return utils::make_unique_ptr<RandomAccessOpenList<StateOpenListEntry>>(options);
}

unique_ptr<EdgeOpenList>
RandomAccessOpenListFactory::create_edge_open_list() {
    return utils::make_unique_ptr<RandomAccessOpenList<EdgeOpenListEntry>>(options);
}