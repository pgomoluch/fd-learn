#ifndef OPEN_LISTS_RA_OPEN_LIST_FACTORY_H
#define OPEN_LISTS_RA_OPEN_LIST_FACTORY_H

#include "random_access_open_list.h"

#include <memory>


class RAOpenListFactory {
public:
    RAOpenListFactory() = default;
    virtual ~RAOpenListFactory() = default;

    RAOpenListFactory(const RAOpenListFactory &) = delete;

    virtual std::unique_ptr<RAStateOpenList> create_state_open_list() = 0;
    virtual std::unique_ptr<RAEdgeOpenList> create_edge_open_list() = 0;

    /*
      The following template receives manual specializations (in the
      cc file) for the open list types we want to support. It is
      intended for templatized callers, e.g. the constructor of
      AlternationOpenList.
    */
    template<typename T>
    std::unique_ptr<RandomAccessOpenList<T>> create_open_list();
};

#endif
