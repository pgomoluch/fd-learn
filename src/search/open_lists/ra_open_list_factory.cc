#include "ra_open_list_factory.h"

#include "../plugin.h"

using namespace std;


template<>
unique_ptr<RAStateOpenList> RAOpenListFactory::create_open_list() {
    return create_state_open_list();
}

template<>
unique_ptr<RAEdgeOpenList> RAOpenListFactory::create_open_list() {
    return create_edge_open_list();
}


static PluginTypePlugin<RAOpenListFactory> _type_plugin(
    "RandomAccessOpenList",
    // TODO: Replace empty string by synopsis for the wiki page.
    "");
