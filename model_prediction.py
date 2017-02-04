import read_data
from datetime import datetime,timedelta
import station_info
from sklearn.preprocessing import Imputer
import numpy
from sklearn import preprocessing
from sklearn.externals import joblib
import matplotlib.pyplot as plt

start_datetime_str = '2017020400';
end_datetime_str = '2017020412';
data_dir = '/scratch/dataops/CURRENT/cur_awsdata';
analysis_stn_id = 'S43';
fcst_hr = 1;
model_path = 'nowcasting_model_S43_L1h_F20160301_TO_20170201.pkl';
model_affi_path = 'nowcasting_model_affi_S43_L1h_F20160301_TO_20170201.npz';

clf = joblib.load(model_path) ;
model_affi_npzfile = numpy.load(model_affi_path);
training_sample = model_affi_npzfile['training_sample'];

stn_list = station_info.station_no;
stn_latlon_list = station_info.station_latlon;

start_datetime = datetime(int(start_datetime_str[0:4]), int(start_datetime_str[4:6]), int(start_datetime_str[6:8]),int(start_datetime_str[8:10]));
end_datetime = datetime(int(end_datetime_str[0:4]), int(end_datetime_str[4:6]), int(end_datetime_str[6:8]), int(end_datetime_str[8:10]));
cur_datetime = start_datetime;

imp = Imputer(missing_values='NaN', strategy='median', axis=0);
pro_list = [];
valid_datetime_str_list = [];
while cur_datetime <= end_datetime:
    valid_datetime_str_list.append((cur_datetime + timedelta(seconds=fcst_hr*3600.0)).strftime('%m%dT%H'));
    print cur_datetime;
    cur_yyyymmdd_train_str = cur_datetime.strftime('%Y%m%d');
    cur_yyyymmddhh_train_str = cur_datetime.strftime('%Y%m%d%H');
    cur_train_dirpath = data_dir + '/' + cur_yyyymmdd_train_str;
    
    cur_train_3g_datapath = cur_train_dirpath + '/' + '3G_' + cur_yyyymmddhh_train_str + '_00.obs';
    cur_train_gauge_filepath = cur_train_dirpath + '/' + 'TB1_Rain_1H_TOT_' + cur_yyyymmddhh_train_str + '.txt';
    data_list = read_data.read_data_predict(analysis_stn_id, cur_train_3g_datapath, cur_train_gauge_filepath,stn_list);
    if ((numpy.count_nonzero(~numpy.isnan(numpy.asarray(data_list)))/(20.0*7.0)) < 0.65):
        print ' --- data discarded: missing data are more than 65% of the total data amount !';
        cur_datetime = cur_datetime + timedelta(seconds=3600);
        pro_list.append(numpy.NaN);
        continue;
    
    imp.fit(data_list);
    data_list2 = imp.transform(data_list);
    
    data_array = numpy.asarray(data_list2);
    
    if data_array.shape[1] != 7:
        print ' --- data discarded: preprocessing cannot address the data missing issue !';
        cur_datetime = cur_datetime + timedelta(seconds=3600);
        pro_list.append(numpy.NaN);
        continue;  
    
    test_sample = data_array.reshape((1, -1));
    combined_sample = numpy.empty((training_sample.shape[0]+1,140));
    combined_sample[0:training_sample.shape[0],:] = training_sample;
    combined_sample[training_sample.shape[0],:] = test_sample;
    combined_sample_scaled = preprocessing.scale(combined_sample);
    test_sample = combined_sample_scaled[training_sample.shape[0],:];
    test_sample2 = test_sample.reshape((1, -1));
    out = clf.predict(test_sample2);
    if out < 0:
        out = 0.0;
    elif out > 1.0:
        out = 1.0;
    
    pro_list.append(out);
    cur_datetime = cur_datetime + timedelta(seconds=3600);

plt.plot(pro_list,'.-');
plt.xticks(range(0,len(pro_list)),valid_datetime_str_list, rotation='vertical');
plt.xlabel('Valid time (T+1h)');
plt.ylabel('Rainfall probability');
plt.grid();
plt.ylim([-0.05,1.05]);
plt.show();
