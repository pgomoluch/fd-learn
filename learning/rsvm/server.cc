#include "heuristic_server.h"

#include <fstream>
#include <iostream>

#include <dlib/svm.h>

using namespace std;


const char *MODEL_PATH = "model.txt";
const int N_FEATURES = 11;

typedef dlib::matrix<double,N_FEATURES*2,1> sample_type;
typedef dlib::linear_kernel<sample_type> kernel_type;


class Server : public HeuristicServer
{
    dlib::decision_function<kernel_type> model;
    double feature_array[N_FEATURES*2] = {0.0};
public:
    Server(): HeuristicServer(N_FEATURES)
    {
        cout << "Loading model..." << endl;
        ifstream model_file(MODEL_PATH);
        dlib::deserialize(model, model_file);
        model_file.close();
        cout << "Loaded model." << endl;
        
        // Print weights
        for (int i = 0; i < N_FEATURES; ++i)
        {
            double sample_array[N_FEATURES * 2] = {0.0};
            sample_array[i] = 1.0;
            sample_type sample(sample_array);
            cout << "w[" << i << "]: " << model(sample) << endl;
        }
        cout << endl;
    }
protected:
    double evaluate(double state[]) override
    {
        for (int i = 0; i < N_FEATURES; ++i)
            feature_array[i] = state[i];
        sample_type sample(feature_array);
        double result = model(sample);
        return result;
    }
};


int main()
{
    Server server;
    server.serve();
}

