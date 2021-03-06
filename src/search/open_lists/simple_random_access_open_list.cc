#include "simple_random_access_open_list.h"

#include "../globals.h"
#include "../option_parser.h"
#include "../plugin.h"

#include "../utils/collections.h"
#include "../utils/markup.h"
#include "../utils/memory.h"
#include "../utils/rng.h"

#include <functional>
#include <memory>

using namespace std;

//class RandomAccessOpenList<StateOpenListEntry>;

template<class HeapNode>
static void adjust_heap_up(vector<HeapNode> &heap, size_t pos) {
    assert(utils::in_bounds(pos, heap));
    while (pos != 0) {
        size_t parent_pos = (pos - 1) / 2;
        if (heap[pos] > heap[parent_pos]) {
            break;
        }
        swap(heap[pos], heap[parent_pos]);
        pos = parent_pos;
    }
}

template<class Entry>
void SimpleRandomAccessOpenList<Entry>::do_insertion(
    EvaluationContext &eval_context, const Entry &entry) {
    heap.emplace_back(
        next_id++, eval_context.get_heuristic_value(evaluator), entry);
    push_heap(heap.begin(), heap.end(), greater<HeapNode>());
    ++size;
}

template<class Entry>
SimpleRandomAccessOpenList<Entry>::SimpleRandomAccessOpenList(const Options &opts)
    : RandomAccessOpenList<Entry>(opts.get<bool>("pref_only")),
      evaluator(opts.get<ScalarEvaluator *>("eval")),
      epsilon(opts.get<double>("epsilon")),
      size(0),
      next_id(0) {
}

template<class Entry>
Entry SimpleRandomAccessOpenList<Entry>::remove_min(vector<int> *key) {
    assert(size > 0);
    pop_heap(heap.begin(), heap.end(), greater<HeapNode>());
    HeapNode heap_node = heap.back();
    heap.pop_back();
    if (key) {
        assert(key->empty());
        key->push_back(heap_node.h);
    }
    --size;
    return heap_node.entry;
}

template<class Entry>
Entry SimpleRandomAccessOpenList<Entry>::remove_random(vector<int> *key) {
    assert(size > 0);
    
    int pos = (*g_rng())(size);
    heap[pos].h = numeric_limits<int>::min();
    adjust_heap_up(heap, pos);

    return remove_min(key);
}

template<class Entry>
Entry SimpleRandomAccessOpenList<Entry>::remove_epsilon(vector<int> *key) {
    if ((*g_rng())() < epsilon)
        return remove_random(key);
    return remove_min(key);
}

template<class Entry>
bool SimpleRandomAccessOpenList<Entry>::is_dead_end(
    EvaluationContext &eval_context) const {
    return eval_context.is_heuristic_infinite(evaluator);
}

template<class Entry>
bool SimpleRandomAccessOpenList<Entry>::is_reliable_dead_end(
    EvaluationContext &eval_context) const {
    return is_dead_end(eval_context) && evaluator->dead_ends_are_reliable();
}

template<class Entry>
void SimpleRandomAccessOpenList<Entry>::get_involved_heuristics(set<Heuristic *> &hset) {
    evaluator->get_involved_heuristics(hset);
}

template<class Entry>
bool SimpleRandomAccessOpenList<Entry>::empty() const {
    return size == 0;
}

template<class Entry>
void SimpleRandomAccessOpenList<Entry>::clear() {
    heap.clear();
    size = 0;
    next_id = 0;
}

SimpleRandomAccessOpenListFactory::SimpleRandomAccessOpenListFactory(
    const Options &options)
    : options(options) {
}

unique_ptr<RAStateOpenList>
SimpleRandomAccessOpenListFactory::create_state_open_list() {
    return utils::make_unique_ptr<SimpleRandomAccessOpenList<StateOpenListEntry>>(options);
}

unique_ptr<RAEdgeOpenList>
SimpleRandomAccessOpenListFactory::create_edge_open_list() {
    return utils::make_unique_ptr<SimpleRandomAccessOpenList<EdgeOpenListEntry>>(options);
}

static shared_ptr<RAOpenListFactory> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "Random access open list",
        "Enables retiving random entries. "
        "Based on the EpsilonGreedyOpenList.");
    parser.add_option<ScalarEvaluator *>("eval", "scalar evaluator");
    parser.add_option<bool>(
        "pref_only",
        "insert only nodes generated by preferred operators", "false");
    parser.add_option<double>(
        "epsilon",
        "probability for choosing the next entry randomly",
        "0.2",
        Bounds("0.0", "1.0"));

    Options opts = parser.parse();
    if (parser.dry_run()) {
        return nullptr;
    } else {
        return make_shared<SimpleRandomAccessOpenListFactory>(opts);
    }
}

static PluginShared<RAOpenListFactory> _plugin("simple_random_access_open_list", _parse);

template class SimpleRandomAccessOpenList<StateOpenListEntry>;