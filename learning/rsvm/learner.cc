#include <dlib/svm.h>
#include <dirent.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace dlib;


const int N_FEATURES = 11;

const int FEATURE_IDS[] = {5,8,10,11,12,13,14,15,16,17,18}; // FF DI
//const int FEATURE_IDS[] = {5,9,19,20,21,22,23,24,25,26,27}; // CEA DI
//const int FEATURE_IDS[] = {5,8,9,10,11,12,13,14,15,16,17,18}; // FF DI + CEA

//const int FEATURE_IDS[] = {5, 8, 13, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}; // FF DD Transport / No-mystery
//const int FEATURE_IDS[] = {5, 8, 13, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}; // FF DD Parking
//const int FEATURE_IDS[] = {5, 8, 13, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69}; // FF DD Elevators

//const int FEATURE_IDS[] = {5, 9, 22, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51}; // CEA DD Transport / No-mystery
//const int FEATURE_IDS[] = {5, 9, 22, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}; // CEA DD Parking
//const int FEATURE_IDS[] = {5, 9, 22, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81}; // CEA DD Elevators

const double C = 1.0;
const std::vector<string> DATA_PATHS = {
    "../data/transport1-10-1000-2-100-2-4-B/",
    "../data/transport-like-D12/",
    "../data/transport-like-D22/",
    "../data/park-4-6-B/",
    "../data/park-5-7-B/",
    "../data/park-5-8-B/",
    "../data/park-6-8-B/",
    "../data/elevators-12-3/",
    "../data/no-mystery-like-M22/"
    //
    //"../data/14C-parking-bfs-ffu/",
    //"../data/11C-elevators-bfs-ffu/",
    //"../data/11C-nomystery-bfs-ffu/"
    //"../data/11C-floortile-bfs-ffu/"
};
const int DATA_LIMITS[] = {7000, 7000, 7000, 5000, 5000, 5000, 5000, 20000, 20000};
const string MODEL_FILE = "model.txt";


typedef matrix<double,N_FEATURES*2,1> sample_type;
typedef linear_kernel<sample_type> kernel_type;

//std::vector<std::vector<double>> global_features;

void load_data(ranking_pair<sample_type> &data);
double point_rank(decision_function<kernel_type> &rank, std::vector<double> point);
double pair_rank(decision_function<kernel_type> &rank, std::vector<double> point1, std::vector<double> point2);

int main()
{
    cout << "RSVM" << endl;
    
    ranking_pair<sample_type> data;
    load_data(data);
    
    svm_rank_trainer<kernel_type> trainer(C);
    cout << "\"Relevant\" samples: " << data.relevant.size() << endl;
    cout << "\"Nonrelevant\" samples: " << data.nonrelevant.size() << endl;
    cout << " C = " << trainer.get_c() << endl;
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
    
    
    for (int i = 0; i < N_FEATURES; ++i)
    {
        double sample_array[N_FEATURES * 2] = {0.0};
        sample_array[i] = 10.0;
        sample_type sample(sample_array);
        cout << "w_" << FEATURE_IDS[i] << ": " << rank(sample) / 10.0 << " ";
        
        double sample_array2[N_FEATURES * 2] = {0.0};
        sample_array2[N_FEATURES+i] = 20.0;
        sample_type sample2(sample_array2);
        cout << "(" << rank(sample2) / -20.0 << ")" << endl;
    }
    
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
    int sample_count = 0;
    //for (const string &path: DATA_PATHS)
    for (int d = 0; d < DATA_PATHS.size(); ++d)
    {
        int batch_sample_count = 0;
        int batch_pair_count = 0;
        int batch_problem_count = 0;
        std::vector<string> filenames;
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir((DATA_PATHS[d] + "features/").c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
                filenames.push_back(ent->d_name);
            closedir(dir);
        }
        
        sort(filenames.begin(), filenames.end());
        
        bool false_dead_end;
        for (string f: filenames)
        {
            false_dead_end = false;
            if (batch_sample_count > DATA_LIMITS[d])
                break;
            
            ifstream feature_file(DATA_PATHS[d] + "features/" + f);
            ifstream label_file(DATA_PATHS[d] + "labels/" + f);
            
            if(!feature_file || !label_file)
            {
                cout << "Problem reading file " << f << endl;
                feature_file.close();
                label_file.close();
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
                

                for (double d: record)
                {
                    if (d > 10000000.0)
                        false_dead_end = true;
                }
                
                //global_features.push_back(record);
                //++sample_count;
                //++batch_sample_count;
            }
            
            feature_file.close();
            label_file.close();
            ++batch_problem_count;
            
            if (false_dead_end) continue;
            
            sample_count += labels.size();
            batch_sample_count += labels.size();
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
                    ++batch_pair_count;
                }
            }
        }
        cout << "Data points from " << DATA_PATHS[d] << ": " << batch_sample_count
            << ". Pairs: " << batch_pair_count << ". Problems: " << batch_problem_count << endl;
    }
    cout << "Total data points: " << sample_count << endl;
}
