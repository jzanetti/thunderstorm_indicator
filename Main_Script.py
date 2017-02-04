###
#sijin
###
from datetime import datetime,timedelta
import os
import read_data
import station_info
from sklearn.preprocessing import Imputer
import numpy
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.externals import joblib

start_datetime_str = '20160301';
end_datetime_str = '20170201';
analysis_stn_id = 'S43';
fcst_length = 2;

run_test_forecasts = False;
test_forecast_percentage = 0.01;
plot_gauge_map = False;

data_dir = '/scratch/dataops/CURRENT/cur_awsdata';
stn_list = station_info.station_no;
stn_latlon_list = station_info.station_latlon;
if plot_gauge_map:
    station_info.plot_gauges(stn_latlon_list,stn_list,analysis_stn_id);

start_datetime = datetime(int(start_datetime_str[0:4]), int(start_datetime_str[4:6]), int(start_datetime_str[6:8]));
end_datetime = datetime(int(end_datetime_str[0:4]), int(end_datetime_str[4:6]), int(end_datetime_str[6:8]));

cur_datetime = start_datetime;

training_sample_data_list = [];
training_truth_data_list = [];
training_truth_data_list_real = [];
datetime_str_list = [];
imp = Imputer(missing_values='NaN', strategy='median', axis=0);
hour_diff = int((end_datetime - start_datetime).total_seconds()/3600.0 + 1);
training_sample_data_array = numpy.empty((hour_diff,20,7));
training_sample_data_array[:] = numpy.NaN;

i_hour = 0;
while cur_datetime <= end_datetime:
    print cur_datetime;
    
    cur_yyyymmdd_train_str = cur_datetime.strftime('%Y%m%d');
    cur_yyyymmddhh_train_str = cur_datetime.strftime('%Y%m%d%H');
    cur_train_dirpath = data_dir + '/' + cur_yyyymmdd_train_str;
    
    cur_yyyymmdd_truth_str = (cur_datetime+ timedelta(seconds=3600*fcst_length)).strftime('%Y%m%d');
    cur_yyyymmddhh_truth_str = (cur_datetime + timedelta(seconds=3600*fcst_length)).strftime('%Y%m%d%H');
    cur_analysis_gauge_dirpath = data_dir + '/' + cur_yyyymmdd_truth_str;
    
    cur_train_3g_datapath = cur_train_dirpath + '/' + '3G_' + cur_yyyymmddhh_train_str + '_00.obs';
    cur_train_gauge_filepath = cur_train_dirpath + '/' + 'TB1_Rain_1H_TOT_' + cur_yyyymmddhh_train_str + '.txt';
    cur_analysis_gauge_filepath = cur_analysis_gauge_dirpath + '/' + 'TB1_Rain_1H_TOT_' + cur_yyyymmddhh_truth_str + '.txt';
    
    if (os.path.exists(cur_train_3g_datapath) == False) or (os.path.exists(cur_train_gauge_filepath) == False) or (os.path.exists(cur_analysis_gauge_filepath) == False):
        cur_datetime = cur_datetime + timedelta(seconds=3600);
        continue;
    
    data_list,analysis_rainfall = read_data.read_data(cur_train_3g_datapath,cur_train_gauge_filepath,cur_analysis_gauge_filepath,stn_list, stn_latlon_list,\
                                                      analysis_stn_id);
    
    data_array = numpy.asarray(data_list);
    if ((numpy.count_nonzero(~numpy.isnan(numpy.asarray(data_list)))/(20.0*7.0)) < 0.75) or (numpy.isnan(analysis_rainfall) == True):
        print ' --- data discarded: missing data are more than 75% of the total data amount !';
        cur_datetime = cur_datetime + timedelta(seconds=3600);
        continue;
    imp.fit(data_list);
    data_list2 = imp.transform(data_list);
    
    if numpy.count_nonzero(numpy.isnan(numpy.asarray(data_list2))) > 0:
        print ' --- data discarded: preprocessing cannot address the data missing issue !';
        training_sample_data_array = numpy.delete(training_sample_data_array, -1, 0);
        cur_datetime = cur_datetime + timedelta(seconds=3600);
        continue;
    training_sample_data_array[i_hour,:,:] = data_list2;

    training_truth_data_list_real.append(analysis_rainfall);
    if analysis_rainfall > 0.0:
        training_truth_data_list.append(1.0);
    else:
        training_truth_data_list.append(0.0);
    datetime_str_list.append(cur_datetime.strftime('%Y%m%dT%H'));
    i_hour = i_hour + 1;
    cur_datetime = cur_datetime + timedelta(seconds=3600);

training_sample_data_array = training_sample_data_array[0:i_hour,:,:];
training_truth_data_array = numpy.asarray(training_truth_data_list);
training_truth_data_array_real = numpy.asarray(training_truth_data_list_real);

n_samples = len(training_sample_data_array);
#training_sample = training_sample_data_array;
training_sample = training_sample_data_array.reshape((n_samples, -1))
training_sample_scaled = preprocessing.scale(training_sample);

if run_test_forecasts:
    training_data_percentage = 1.0 - test_forecast_percentage;
    training_sample_scaled2 = training_sample_scaled[0:int(training_sample_scaled.shape[0]*training_data_percentage),:];
    training_truth_data_array2 = training_truth_data_array[0:int(training_sample_scaled.shape[0]*training_data_percentage)];
    testing_sample_scaled = training_sample_scaled[int(training_sample_scaled.shape[0]*training_data_percentage)+1:-1,:];
    testing_truth = training_truth_data_array[int(training_sample_scaled.shape[0]*training_data_percentage)+1:-1];
    testing_truth_real = training_truth_data_array_real[int(training_sample_scaled.shape[0]*training_data_percentage)+1:-1];
    datetime_str = datetime_str_list[int(training_sample_scaled.shape[0]*training_data_percentage)+1:-1];
    number_of_training_dataset = len(training_truth_data_array2);
    number_of_testing_dataset = len(testing_truth_real);
else:
    training_sample_scaled2 = training_sample_scaled;
    training_truth_data_array2 = training_truth_data_array;
    number_of_training_dataset = len(training_truth_data_array2);
    number_of_testing_dataset = None;     

model_data_suffix = analysis_stn_id + '_L' + str(fcst_length) + 'h_F' + start_datetime_str + '_TO_' + end_datetime_str;
clf = MLPRegressor(hidden_layer_sizes=(1500,),verbose=True);
clf.fit(training_sample_scaled2,training_truth_data_array2);
joblib.dump(clf, 'nowcasting_model_' + model_data_suffix +  '.pkl');
if run_test_forecasts:
    out = clf.predict(testing_sample_scaled);
    out[out < 0.0] = 0.0;
    out[out > 99] = numpy.NaN;
    out[out > 1] = 1.0;

numpy.savez('nowcasting_model_affi_' + model_data_suffix, \
            training_sample=training_sample,\
            number_of_training_dataset = number_of_training_dataset, \
            number_of_testing_dataset = number_of_testing_dataset);

print '-------------------------------------------------'
print 'Total sample data size: (' + str(training_sample_data_array.shape[0]) + ', ' \
                           + str(training_sample_data_array.shape[1]) + ')';
print 'Total truth data size: (' + str(training_sample.shape[0]) + ')';
print 'Training sample data size: (' + str(training_sample_scaled2.shape[0]) + ', ' \
                           + str(training_sample_scaled2.shape[1]) + ')';
print 'Training truth data size: (' + str(training_truth_data_array2.shape[0]) + ')';

if run_test_forecasts:
    print 'Testing sample data size: (' + str(testing_sample_scaled.shape[0]) + ', ' \
                           + str(testing_sample_scaled.shape[1]) + ')';
    print 'Training truth data size: (' + str(testing_truth.shape[0]) + ')';
else:
    print 'Testing is switched off';
print '-------------------------------------------------'

if run_test_forecasts:
    fig, ax1 = plt.subplots()
    ax1.plot(out, 'b-')
    ax1.set_xlabel('time')
    plt.xticks(range(0,len(out),24),datetime_str[0:-1:24], rotation='vertical');
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Precipitation probability', color='b');
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx();
    testing_truth_real[testing_truth_real > 0.0] = 1.0;
    testing_truth_real[testing_truth_real == 0.0] = numpy.NaN;
    ax2.plot(testing_truth_real, 'r.')
    ax2.set_ylabel('Observed hourly rainfall (YES/NO)', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout();

    plt.savefig('Testing_' + model_data_suffix + '.png');
    plt.close();

print 'Processes completed !';

