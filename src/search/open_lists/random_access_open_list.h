#ifndef OPEN_LISTS_I_RANDOM_ACCESS_OPEN_LIST_H
#define OPEN_LISTS_I_RANDOM_ACCESS_OPEN_LIST_H

#include "open_list.h"

template<class Entry>
class RandomAccessOpenList : public OpenList<Entry> {
public:
    explicit RandomAccessOpenList(bool preferred_only = false)
        : OpenList<Entry>(preferred_only) {}
    virtual ~RandomAccessOpenList() = default;

    virtual Entry remove_random(std::vector<int> *key = nullptr) = 0;
    virtual Entry remove_epsilon(std::vector<int> *key = nullptr) = 0;
};

using RAStateOpenList = RandomAccessOpenList<StateOpenListEntry>;
using RAEdgeOpenList = RandomAccessOpenList<EdgeOpenListEntry>;

#endif