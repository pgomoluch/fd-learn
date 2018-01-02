#ifndef OPEN_LISTS_RANDOM_ACCESS_OPEN_LIST_H
#define OPEN_LISTS_RANDOM_ACCESS_OPEN_LIST_H

/*
  Open list allowing for random access to stored elements, developed for
  use with the learning search engine (LearningSearch class). Starts as
  a copy of StandardScalarOpenList, but is expected to diverge quickly.
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

#endif