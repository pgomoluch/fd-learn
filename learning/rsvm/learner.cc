#include <dlib/svm.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace dlib;


const int N_FEATURES = 18;
const int FEATURE_IDS[] = {5,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
const int START_FILE_ID = 1100;
const int END_FILE_ID = 1200;//11100;
const string DATA_PATH = "../data/transport1-10-1000-2-100-2-4-C/";
const string MODEL_FILE = "model.txt";


typedef matrix<double,N_FEATURES*2,1> sample_type;
typedef linear_kernel<sample_type> kernel_type;

std::vector<std::vector<double>> global_features;

void load_data(ranking_pair<sample_type> &data);
double point_rank(decision_function<kernel_type> &rank, std::vector<double> point);
double pair_rank(decision_function<kernel_type> &rank, std::vector<double> point1, std::vector<double> point2);

int main()
{
    cout << "RSVM" << endl;
    
    ranking_pair<sample_type> data;
    load_data(data);
    
    svm_rank_trainer<kernel_type> trainer;
    cout << "\"Relevant\" samples: " << data.relevant.size() << endl;
    cout << "\"Nonrelevant\" samples: " << data.nonrelevant.size() << endl;
    decision_function<kernel_type> rank = trainer.train(data);
    
    cout << "(ordering accuracy, mean average precision): " << test_ranking_function(rank, data) << endl;
    
    ofstream model_file(MODEL_FILE);
    serialize(rank, model_file);
    model_file.close();
    
    decision_function<kernel_type> rank2;
    ifstream in_file(MODEL_FILE);
    deserialize(rank2, in_file);
    in_file.close();
    
    cout << "Rank of the first relevant sample: " << rank(data.relevant[0]) << endl;
    cout << "Rank of the first relevant sample (loaded function): " << rank2(data.relevant[0]) << endl;
    
    // Verify the ranks on the first 20 planning states
    //for (int i = 0; i < 20; ++i)
    //    for (int j = 0; j < 20; ++j)
    //    {
    //        double point_i = point_rank(rank, global_features[i]);
    //        double point_j = point_rank(rank, global_features[j]);
    //        cout << pair_rank(rank, global_features[i], global_features[j]) << "  "
    //            << point_i - point_j << "  "
    //            << point_i << "  "
    //            << point_j << endl;
    //    }
    
    return 0;
}

double point_rank(decision_function<kernel_type> &rank, std::vector<double> point)
{
    double feature_array[N_FEATURES * 2];
    for (int i = 0; i < N_FEATURES; ++i)
    {
        feature_array[i] = point[i];
        feature_array[N_FEATURES + i] = 0.0;
    }
    sample_type sample(feature_array);
    return rank(sample);
}

double pair_rank(decision_function<kernel_type> &rank, std::vector<double> point1, std::vector<double> point2)
{
    double feature_array[N_FEATURES * 2];
    for (int i = 0; i < N_FEATURES; ++i)
    {
        feature_array[i] = point1[i];
        feature_array[N_FEATURES + i] = point2[i];
    }
    sample_type sample(feature_array);
    return rank(sample);
}

void load_data(ranking_pair<sample_type> &data)
{
    int missing_files = 0;
    for (int file_id = START_FILE_ID; file_id <= END_FILE_ID; ++file_id)
    {
        
        ifstream feature_file(DATA_PATH + "features/" + to_string(file_id) + ".txt");
        ifstream label_file(DATA_PATH + "labels/" + to_string(file_id) + ".txt");
        
        if(!feature_file || !label_file)
        {
            cout << "Problem reading file " << file_id << endl;
            feature_file.close();
            label_file.close();
            ++missing_files;
            continue;
        }
        
        std::vector<std::vector<double>> features;
        std::vector<double> labels;
        
        string line;
        while(getline(feature_file, line))
        {
            stringstream line_stream(line);
            std::vector<double> raw_record;
            double d;
            while(line_stream >> d)
                raw_record.push_back(d);
            
            std::vector<double> record;
            for (int i = 0; i < N_FEATURES; ++i)
                record.push_back(raw_record[FEATURE_IDS[i]]);
            
            int i;
            label_file >> i;
            
            features.push_back(record);
            labels.push_back((double)i);
            
            global_features.push_back(record);
        }
        
        feature_file.close();
        label_file.close();
        
        // Relies on the ordering of states: from the initial one to the goal
        for (int i = 0; i < labels.size(); ++i)
        {
            for (int j = i+1; j < labels.size(); ++j)
            {
                double array[N_FEATURES * 2];
                double array1[N_FEATURES * 2];
                
                for (int k = 0; k < N_FEATURES; ++k)
                {
                    array1[N_FEATURES + k] = array[k] = features[i][k];
                    array1[k] = array[N_FEATURES + k] = features[j][k];    
                }
                sample_type sample(array);
                sample_type sample1(array1);
                
                data.relevant.push_back(sample);
                data.nonrelevant.push_back(sample1);           
            }
        }
    }
    cout << "Files missing: " << missing_files << endl;
}
