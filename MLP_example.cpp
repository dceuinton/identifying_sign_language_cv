/**
 * How to use it with the Glass.data:
 * g++ MLP_example.cpp -o MLP_example `pkg-config --cflags --libs opencv`
 * 
 * Train:
 * 
 * ./MLP_example -save example.xml -data Glass.data
 * 
 * 
 * Test:
 * 
 * ./MLP_example -load example.xml -data Glass.data
 * 
 */

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

// This function reads data and responses from the file <filename>
static bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses )
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back(buf[0]);
//char test;
//test=buf[0]+65;
//responses.push_back(test);
cout << "responses " << buf[0] << " " ;;//<<  endl;
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
cout << el_ptr << endl;
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save)
{
    int i, nsamples_all = data.rows;
    double train_hr = 0, test_hr = 0;
    int training_correct_predict=0;
    // compute prediction error on training data
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);
cout << "Sample: " << responses.at<int>(i)-48 << " row " << data.row(i) << endl;
        float r = model->predict( sample );
cout << "Predict:  r = " << r << endl;

cout << (int) responses.at<int>(i) << " vs: " << (int)(responses.at<int>(i)-48) << endl;
   if( (int)r == (int)(responses.at<int>(i)-48) ) //prediction is correct
	  training_correct_predict++;
   
    //r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
	
     
        //if( i < ntrain_samples )
        //    train_hr += r;
        //else
        //    test_hr += r;
    }

    //test_hr /= nsamples_all - ntrain_samples;
    //train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;
    printf("ntrain_samples %d training_correct_predict %d \n",ntrain_samples, training_correct_predict);
    if( filename_to_save.empty() )  printf( "\nTest Recognition rate: training set = %.1f%% \n\n", training_correct_predict*100.0/ntrain_samples);


    if( !filename_to_save.empty() )
    {
        model->save( filename_to_save );
    }
/*************   Example of how to predict a single sample ************************/   
// Use that for the assignment3, for every frame after computing the features, r is the prediction given the features listed in this format
    // Mat sample = data.row(i);

    float r;
    
    // 1
    // Mat sample0 = (Mat_<float>(1, 9) << 0.483537,0.0897202,0.0896418,0.0904506,0.0587337,0.027807,0.00611194,0.0210563,0.0212424);
    Mat sample0 = (Mat_<float>(1, 9) << 0.192112,0.209336,0.0675378,0.0860401,0.0592248,0.0402964,0.0492288,0.0460786,0.0377082);
    r = model->predict(sample0);
    printf("Predicted: %f\n", r);

    // 2 
    Mat sample1 = (Mat_<float>(1, 9) << 0.288093,0.263243,0.150342,0.13571,0.0671558,0.0666056,0.0320719,0.0334513,0.0270351);
    r = model->predict(sample1);
    printf("Predicted: %f\n", r);

    // 3
    Mat sample2 = (Mat_<float>(1, 9) << 0.863456,0.546805,0.379552,0.186811,0.167384,0.0793804,0.0974595,0.0781477,0.0230621);
    r = model->predict(sample2);
    printf("Predicted: %f\n", r);

    // 4
    Mat sample3 = (Mat_<float>(1, 9) << 0.67012,0.196405,0.460634,0.20463,0.116233,0.0743352,0.0783094,0.0690908,0.0195137);
    r = model->predict(sample3);
    printf("Predicted: %f\n", r);

    // 5
    Mat sample4 = (Mat_<float>(1, 9) << 1.20807,0.374261,0.27939,0.320042,0.273963,0.15068,0.119287,0.129897,0.061082);
    r = model->predict(sample4);
    printf("Predicted: %f\n", r);

    // 6
    Mat sample5 = (Mat_<float>(1, 9) << 0.583613,0.341947,0.123211,0.163423,0.101956,0.184059,0.230072,0.186048,0.0613964);
    r = model->predict(sample5);
    printf("Predicted: %f\n", r);

    // Mat sample1 = (Mat_<float>(1,14) << 0.0952995,0.169177,0.0592047,0.0618227,0.0319202,0.0125807,0.0169665,0.0175996,0.0172938,0.0214138,0.0122307,0.0105928,0.00865961,0.00718095);// 1
    // float r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    // sample1 = (Mat_<float>(1,14) << 0.394995,0.323635,0.164296,0.0473424,0.0432019,0.0361735,0.0548237,0.037759,0.0335067,0.0148624,0.0180722,0.0240899,0.0175381,0.0110956);//2
    // r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    // sample1 = (Mat_<float>(1,14) << 0.529972,0.370616,0.304438,0.200361,0.0747549,0.072658,0.0423362,0.0658548,0.042409,0.0309975,0.043568,0.0277283,0.0177653,0.0245778);//3
    // r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    // sample1 = (Mat_<float>(1,14) << 0.536639,0.131383,0.239704,0.289744,0.214504,0.157104,0.113756,0.0488208,0.0335598,0.0304096,0.0277343,0.0321744,0.031369,0.0314621);//4
    // r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    // sample1 = (Mat_<float>(1,14) << 0.778576,0.354334,0.183864,0.209804,0.324091,0.247657,0.195179,0.111713,0.0603599,0.0429004,0.0505883,0.0515825,0.0534233,0.0213703);//5
    // r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    // sample1 = (Mat_<float>(1,14) << 0.644056,0.338977,0.323534,0.134752,0.259656,0.340568,0.192196,0.162745,0.0409341,0.0423454,0.042068,0.0310394,0.0353203,0.0501668);//6
    // r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    // sample1 = (Mat_<float>(1,14) << 0.479818,0.269357,0.199477,0.185035,0.298081,0.247759,0.0531962,0.0653802,0.0512507,0.0244664,0.0371893,0.0537663,0.0542912,0.031215);//7
    // r = model->predict( sample1 );
    // cout << "Prediction: " << r << endl;
    
/**********************************************************************/    
    
}



static bool
build_mlp_classifier( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load )
{
    const int class_count = 24;//CLASSES
    Mat data;
    Mat responses;

    bool ok = read_num_class_data( data_filename, 9, &data, &responses );//third parameter: FEATURES
    if( !ok )
        return ok;

    Ptr<ANN_MLP> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*1.0);//SPLIT

    // Create or load MLP classifier
    if( !filename_to_load.empty() )
    {
        model = load_classifier<ANN_MLP>(filename_to_load);
        if( model.empty() )
            return false;
        // ntrain_samples = 0;
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // MLP does not support categorical variables by explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of <class_count> components for training and,
        // therefore, MLP will give us a vector of "probabilities" at the
        // prediction stage
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Mat train_data = data.rowRange(0, ntrain_samples);
        Mat train_responses = Mat::zeros( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        cout << "Unrolling the responses...\n";
        for( int i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = responses.at<int>(i) - 48;// - 'A'; //change to numerical classes, still they read as chars
            cout << "labels " << cls_label << endl;
            train_responses.at<float>(i, cls_label) = 1.f;
        }

        // 2. train classifier
        int layer_sz[] = { data.cols, 100, 100, class_count };
        // int layer_sz[] = { data.cols, 100, class_count };
	cout <<  " sizeof layer_sz " << sizeof(layer_sz) << " sizeof layer_sz[0]) " << sizeof(layer_sz[0]) << endl;
        int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
	cout << " nlayers  " << nlayers << endl;
        Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );

#if 1
        int method = ANN_MLP::BACKPROP;
        double method_param = 0.003;
        int max_iter = 600;
#else
        int method = ANN_MLP::RPROP;
        double method_param = 0.1;
        int max_iter = 1000;
#endif

        Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

        cout << "Training the classifier (may take a few minutes)...\n";
        model = ANN_MLP::create();
        model->setLayerSizes(layer_sizes);
        model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
        model->setTermCriteria(TC(max_iter,0));
        model->setTrainMethod(method, method_param);
        model->train(tdata);
        cout << endl;
    }

    //test_and_save_classifier(model, data, responses, ntrain_samples, 'A', filename_to_save);
    test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
    // printf("DALES DEBUGGING ------- Data.cols: %i\n", (int) data.cols);
    return true;
}


int main( int argc, char *argv[] )
{
    string filename_to_save = "";
    string filename_to_load = "";
    string data_filename = "letter-recognition.data";
    int method = 0;

    int i;
    for( i = 1; i < argc; i++ )
    {
        if( strcmp(argv[i],"-data") == 0 ) // flag "-data letter_recognition.xml"
        {
            i++;
            data_filename = argv[i];
        }
        else if( strcmp(argv[i],"-save") == 0 ) // flag "-save filename.xml"
        {
            i++;
            filename_to_save = argv[i];
	    cout << "filename to save is "<< filename_to_save << endl;
        }
        else if( strcmp(argv[i],"-load") == 0) // flag "-load filename.xml"
        {
            i++;
            filename_to_load = argv[i];
        }
    }
    build_mlp_classifier( data_filename, filename_to_save, filename_to_load );
}
