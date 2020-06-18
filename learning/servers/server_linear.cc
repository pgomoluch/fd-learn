#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

using namespace std;

const char *socket_path = "/tmp/fd-learn-socket";
const char *model_path = "model.txt";
const int n_features = 11;

bool load_model(const char *file, vector<double> &weights, double &intercept);
double evaluate(vector<double> &weights, double intercept, double features[]);
void terminate(int code, const char *msg);

int main()
{
    double intercept;
    vector<double> weights;
    char buf[1000];
    
    cout << "Loading model..." << endl;
    if(!load_model(model_path, weights, intercept))
    {
        cout << "Failed to load the model" << endl;
        return 1;
    }
    cout << "Loaded model (" << weights.size() << " weights)." << endl;
    
    
    int fd;
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(fd == -1)
        return 2;
    
    int reuse_addr = 1;
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
    
    while(true)
    {
        int cl = accept(fd, NULL, NULL);
        if(cl == -1)
        {
            cout << "Error on accept" << endl;
            return 5;
        }

        int rc;
        while(true)
        {
            bool closed_by_client = false;
            int rc;
            int n_read = 0;
            while(n_read < weights.size()*sizeof(double))
            {
                rc = read(cl, buf+n_read, sizeof(buf));
                if(rc == 0)
                {
                    //terminate(0, "Connection closed by client");
                    closed_by_client = true;
                    break;
                }
                n_read += rc;
            }
            if(closed_by_client)
                break;
            double *features = (double*)buf;
            double result = evaluate(weights, intercept, features);
            int n_written = 0;
            while(n_written < sizeof(double))
            {
                rc = write(cl, &result, sizeof(double));
                n_written += rc;
            }
        }
        // dead code for now
        close(cl);
    }
    unlink(socket_path);
    cout << "Terminating successfully..." << endl;
    return 0;
}

void terminate(int code, const char *msg)
{
    unlink(socket_path);
    cout << msg << endl;
    exit(code);
}

bool load_model(const char *file, vector<double> &weights, double &intercept)
{
    ifstream in(file);
    if(!in)
        return false;
    in >> intercept;
    double d;
    while(in >> d)
        weights.push_back(d);
    in.close();
    return true;
}

double evaluate(vector<double> &weights, double intercept, double features[])
{
    double result = intercept;
    for(int i=0; i<weights.size(); ++i)
        result += weights[i] * features[i];
    return result;
}

