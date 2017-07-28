#include "heuristic_server.h"

#include <csignal>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

using namespace std;

const char *socket_path = "/tmp/fd-learn-socket";

HeuristicServer::HeuristicServer(int n_features) : n_features(n_features)
{
    cout << "n_features = " << n_features << endl;
}

int HeuristicServer::serve()
{
    char buf[1000];
    
    signal(SIGPIPE, SIG_IGN);
    
    int fd;
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(fd == -1)
        return 2;
    
    //int reuse_addr = 1;
    //setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(int));
    
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, socket_path);
    
    cout << "Binding..." << endl;
    if(bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1)
    {
        cout << "Failed to bind." << endl;   
        return 3;
    }
    
    if(listen(fd, 5) == -1) {
        cout << "Failed to listen" << endl;
        return 4;
    }
    cout << "Listening at " << socket_path << endl;
    
    int connection_count = 0;
    while (true)
    {
        int cl = accept(fd, NULL, NULL);
        if(cl == -1)
        {
            cout << "Error on accept" << endl;
            return 5;
        }
        
        connection_count += 1;
        cout << "Accepted a new connection. (" << connection_count << ")" << endl;
        double max_result = 0.0;
        while(true)
        {
            int rc;
            int n_read = 0;
            bool connection_terminated = false;
            while(n_read < n_features * sizeof(double))
            {
                rc = read(cl, buf+n_read, sizeof(buf));
                if(rc == 0)
                {
                    connection_terminated = true;
                    break;
                }
                n_read += rc;
            }
            if (connection_terminated)
            {
                cout << "Connection terminated by client." << endl;
                break;
            }
            double *features = (double*)buf;
            //for (int i = 0; i < n_features; ++i)
            //    cout << features[i] << " ";
            //cout << endl;
            double result = evaluate(features);
            int n_written = 0;
            while(n_written < sizeof(double))
            {
                rc = write(cl, ((char*)&result) + n_written, sizeof(double) - n_written);
                n_written += rc;
            }    
        }
    }  
}

