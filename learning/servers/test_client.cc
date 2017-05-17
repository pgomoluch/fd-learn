#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

using namespace std;

const char *socket_path = "fd-learn-socket";

int main()
{
    struct sockaddr_un addr;
    char buf[100];
    double vec[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(fd == -1)
    {
        cout << "Socket error.";
        return 2;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, socket_path);
    
    if(connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        cout << "Error connecting." << endl;
        return 1;
    }
    
    for(int i=0; i<5; i++)
    {
        vec[i] += (i+1) * 1.1;
        write(fd, vec, sizeof(vec));
        
        int n_read = 0;
        while(n_read < sizeof(double))
        {
            int rc = 0;
            rc = read(fd, buf+n_read, sizeof(double)); // will not get more anyway
            n_read += rc;
        }
        
        double response = *((double*)buf);
        cout << "Evaluation: " << response << endl;
    }
    
    close(fd);
    return 0;
}
