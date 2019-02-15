#include "external_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

using namespace std;

namespace external_heuristic {

const int INITIAL_STATE_VALUE = 1000000;
const char *socket_path = "/tmp/fd-learn-socket";

ExternalHeuristic::ExternalHeuristic(const options::Options &options)
    : Heuristic(options), is_scaling_initialized(false) {
    cout << "Initializing external heuristic..." << endl;
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(fd == -1)
    {
        cout << "Error creating a socket." << endl;
        throw 77;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, socket_path);
    if(connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        cout << "Error connecting." << endl;
        throw 77;
    }
}

ExternalHeuristic::~ExternalHeuristic() {
    close(fd);
}

int ExternalHeuristic::compute_heuristic(const GlobalState &global_state) {
    char buf[100];
    
    auto features = state_encoder.encode(global_state);

    if (state_encoder.is_infinite())
        return EvaluationResult::INFTY;

    unsigned n_written = write(fd, &(features[0]), features.size()*sizeof(double));
    if(n_written < features.size()*sizeof(double)) //TMP
        throw 77;
    
    unsigned n_read = 0;
    while(n_read < sizeof(double))
    {
        int rc = 0;
        rc = read(fd, buf+n_read, sizeof(double)); // will not get more anyway
        n_read += rc;
    }    
    double result;// = *((double*)buf);
    memcpy(&result, buf, sizeof(double));

    for (auto op: state_encoder.get_preferred_operators())
        set_preferred(op);

    //if(result > 2000000000.0) // I would comapre to max(int), but precision issues?
    //    return 2147483647;

    /*if (is_scaling_initialized)
        result *= scaling_factor;
    else
    {
        scaling_factor = INITIAL_STATE_VALUE / result;
        result = INITIAL_STATE_VALUE;
        is_scaling_initialized = true;
    }*/
    
    return result;
}

static Heuristic *_parse(OptionParser &parser) {
    parser.document_synopsis("External heuristic", "");
    parser.document_language_support("action costs", "ignored by design");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support("axioms", "supported");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "no");
    parser.document_property("preferred operators", "yes");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new ExternalHeuristic(opts);
}

static Plugin<Heuristic> _plugin("external", _parse);

}
