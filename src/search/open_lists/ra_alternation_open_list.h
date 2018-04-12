#ifndef OPEN_LISTS_RA_ALTERNATION_OPEN_LIST_H
#define OPEN_LISTS_RA_ALTERNATION_OPEN_LIST_H

#include "ra_open_list_factory.h"

#include "../option_parser_util.h"


class RAAlternationOpenListFactory : public RAOpenListFactory {
    Options options;
public:
    explicit RAAlternationOpenListFactory(const Options &options);
    virtual ~RAAlternationOpenListFactory() override = default;

    virtual std::unique_ptr<RAStateOpenList> create_state_open_list() override;
    virtual std::unique_ptr<RAEdgeOpenList> create_edge_open_list() override;
};

#endif
