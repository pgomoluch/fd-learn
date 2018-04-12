#include "ra_alternation_open_list.h"

#include "random_access_open_list.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../utils/memory.h"
#include "../utils/system.h"

#include <cassert>
#include <memory>
#include <vector>

using namespace std;
using utils::ExitCode;


template<class Entry>
class RAAlternationOpenList : public RandomAccessOpenList<Entry> {
    vector<unique_ptr<RandomAccessOpenList<Entry>>> open_lists;
    vector<int> priorities;

    const int boost_amount;
protected:
    virtual void do_insertion(EvaluationContext &eval_context,
                              const Entry &entry) override;

public:
    explicit RAAlternationOpenList(const Options &opts);
    virtual ~RAAlternationOpenList() override = default;

    virtual Entry remove_min(vector<int> *key = nullptr) override;
    virtual Entry remove_random(vector<int> *key = nullptr) override;
    virtual Entry remove_epsilon(vector<int> *key = nullptr) override;
    virtual bool empty() const override;
    virtual void clear() override;
    virtual void boost_preferred() override;
    virtual void get_involved_heuristics(set<Heuristic *> &hset) override;
    virtual bool is_dead_end(
        EvaluationContext &eval_context) const override;
    virtual bool is_reliable_dead_end(
        EvaluationContext &eval_context) const override;
};


template<class Entry>
RAAlternationOpenList<Entry>::RAAlternationOpenList(const Options &opts)
    : boost_amount(opts.get<int>("boost")) {
    vector<shared_ptr<RAOpenListFactory>> open_list_factories(
        opts.get_list<shared_ptr<RAOpenListFactory>>("sublists"));
    open_lists.reserve(open_list_factories.size());
    for (const auto &factory : open_list_factories)
        open_lists.push_back(factory->create_open_list<Entry>());

    priorities.resize(open_lists.size(), 0);
}

template<class Entry>
void RAAlternationOpenList<Entry>::do_insertion(
    EvaluationContext &eval_context, const Entry &entry) {
    for (const auto &sublist : open_lists)
        sublist->insert(eval_context, entry);
}

template<class Entry>
Entry RAAlternationOpenList<Entry>::remove_min(vector<int> *key) {
    if (key) {
        cerr << "not implemented -- see msg639 in the tracker" << endl;
        utils::exit_with(ExitCode::UNSUPPORTED);
    }
    int best = -1;
    for (size_t i = 0; i < open_lists.size(); ++i) {
        if (!open_lists[i]->empty() &&
            (best == -1 || priorities[i] < priorities[best])) {
            best = i;
        }
    }
    assert(best != -1);
    const auto &best_list = open_lists[best];
    assert(!best_list->empty());
    ++priorities[best];
    return best_list->remove_min(nullptr);
}

// TODO: Replace dummy implementations of remove_random and remove_epsilon.
template<class Entry>
Entry RAAlternationOpenList<Entry>::remove_random(vector<int> *key) {
    return remove_min(key);
}

template<class Entry>
Entry RAAlternationOpenList<Entry>::remove_epsilon(vector<int> *key) {
    return remove_min(key);
}

template<class Entry>
bool RAAlternationOpenList<Entry>::empty() const {
    for (const auto &sublist : open_lists)
        if (!sublist->empty())
            return false;
    return true;
}

template<class Entry>
void RAAlternationOpenList<Entry>::clear() {
    for (const auto &sublist : open_lists)
        sublist->clear();
}

template<class Entry>
void RAAlternationOpenList<Entry>::boost_preferred() {
    for (size_t i = 0; i < open_lists.size(); ++i)
        if (open_lists[i]->only_contains_preferred_entries())
            priorities[i] -= boost_amount;
}

template<class Entry>
void RAAlternationOpenList<Entry>::get_involved_heuristics(
    set<Heuristic *> &hset) {
    for (const auto &sublist : open_lists)
        sublist->get_involved_heuristics(hset);
}

template<class Entry>
bool RAAlternationOpenList<Entry>::is_dead_end(
    EvaluationContext &eval_context) const {
    // If one sublist is sure we have a dead end, return true.
    if (is_reliable_dead_end(eval_context))
        return true;
    // Otherwise, return true if all sublists agree this is a dead-end.
    for (const auto &sublist : open_lists)
        if (!sublist->is_dead_end(eval_context))
            return false;
    return true;
}

template<class Entry>
bool RAAlternationOpenList<Entry>::is_reliable_dead_end(
    EvaluationContext &eval_context) const {
    for (const auto &sublist : open_lists)
        if (sublist->is_reliable_dead_end(eval_context))
            return true;
    return false;
}


RAAlternationOpenListFactory::RAAlternationOpenListFactory(const Options &options)
    : options(options) {
}

unique_ptr<RAStateOpenList>
RAAlternationOpenListFactory::create_state_open_list() {
    return utils::make_unique_ptr<RAAlternationOpenList<StateOpenListEntry>>(options);
}

unique_ptr<RAEdgeOpenList>
RAAlternationOpenListFactory::create_edge_open_list() {
    return utils::make_unique_ptr<RAAlternationOpenList<EdgeOpenListEntry>>(options);
}

static shared_ptr<RAOpenListFactory> _parse(OptionParser &parser) {
    parser.document_synopsis("Random access alternation open list",
                             "alternates between several open lists.");
    parser.add_list_option<shared_ptr<RAOpenListFactory>>(
        "sublists",
        "open lists between which this one alternates");
    parser.add_option<int>(
        "boost",
        "boost value for contained open lists that are restricted "
        "to preferred successors",
        "0");

    Options opts = parser.parse();
    opts.verify_list_non_empty<shared_ptr<RAOpenListFactory>>("sublists");
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<RAAlternationOpenListFactory>(opts);
}

static PluginShared<RAOpenListFactory> _plugin("ra_alt", _parse);
