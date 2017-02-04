import numpy
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from metpy.calc.basic import get_wind_components
from metpy.calc.kinematics import h_convergence
from metpy.units import units

def analysis_point_grid(analysis_stn_id,stn_list, stn_lat_list, stn_lon_list):
    delta_degree = 0.5;
    stn_index = stn_list.index(analysis_stn_id);
    analysis_stn_lat = stn_lat_list[stn_index];
    analysis_stn_lon = stn_lon_list[stn_index];
    analysis_lat, analysis_lon = numpy.mgrid[(analysis_stn_lat - delta_degree):(analysis_stn_lat + delta_degree):50j, (analysis_stn_lon - delta_degree):(analysis_stn_lon + delta_degree):50j]
    return analysis_lat,analysis_lon;

def read_latlon(sample_data):
    stn_list = [];
    stn_lat_list = [];
    stn_lon_list = [];
    with open(sample_data) as f:
        for line in f:
            processed_line1 = line.split(' ');
            processed_line2 = filter(None, processed_line1);
            stn_list.append(processed_line2[0]);
            stn_lon_list.append(float(processed_line2[1]));
            stn_lat_list.append(float(processed_line2[2]));
    return stn_list, stn_lat_list, stn_lon_list;

def read_data(datapath_3g,datapath_gauge,cur_analysis_gauge_filepath,stn_list, stn_latlon_list,\
              analysis_stn_id):
    data_list = [];
    stn_num = 0;
    analysis_rainfall = numpy.NaN;
    # analysis gauge
    with open(cur_analysis_gauge_filepath) as f_gauge:
        for line in f_gauge:
            processed_line1 = line.split(' ');
            processed_line2 = filter(None, processed_line1);
            if processed_line2[0].strip() != analysis_stn_id:
                continue;
            analysis_rainfall = float(processed_line2[3]);
    
    for i_stn in range(0,len(stn_list)):
        cur_stn = stn_list[i_stn];
        cur_data_list = [numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN];
        # read gauge
        with open(datapath_gauge) as f_gauge:
            for line in f_gauge:
                processed_line1 = line.split(' ');
                processed_line2 = filter(None, processed_line1);
                if processed_line2[0].strip() != cur_stn:
                    continue;
                cur_data_list[6] = float(processed_line2[3]);
    
        # read 3g
        with open(datapath_3g) as f_3g:
            for line in f_3g:
                processed_line1 = line.split(',');
                processed_line2 = filter(None, processed_line1);
                if processed_line2[1].strip() != cur_stn:
                    continue;
                stn_num = stn_num + 1;
                if any('DBT' in item for item in processed_line2):
                    dry_bulb_temperature = float(processed_line2[3]);
                    cur_data_list[0] = dry_bulb_temperature;
                if any('DPT' in item for item in processed_line2):
                    dew_point_temperature = float(processed_line2[3]);
                    cur_data_list[1] = dew_point_temperature;
                if any('QFE' in item for item in processed_line2): 
                    atmospheric_pressure_at_aero = float(processed_line2[3]);
                    cur_data_list[2] = atmospheric_pressure_at_aero;
                if any('RH' in item for item in processed_line2): 
                    relative_humidity = float(processed_line2[3]);  
                    cur_data_list[3] = relative_humidity; 
                if any('Wind Dir' in item for item in processed_line2): 
                    wind_dir = float(processed_line2[3]);  
                    cur_data_list[4] = wind_dir; 
                if any('Wind Speed' in item for item in processed_line2): 
                    wind_spd = float(processed_line2[3]); 
                    cur_data_list[5] = wind_spd;
        
        if cur_data_list[0] == '':
            print 'Stn: ' + str(cur_stn) + ' is not available !';
            analysis_rainfall = numpy.NaN;
         
        data_list.append(cur_data_list)
    
    return data_list,analysis_rainfall;
    
def read_data_predict(analysis_stn_id,cur_train_3g_datapath,cur_train_gauge_filepath,stn_list):
    data_list = [];
    stn_num = 0;
    
    for i_stn in range(0,len(stn_list)):
        cur_stn = stn_list[i_stn];
        cur_data_list = [numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN,numpy.NaN];
        # read gauge
        with open(cur_train_gauge_filepath) as f_gauge:
            for line in f_gauge:
                processed_line1 = line.split(' ');
                processed_line2 = filter(None, processed_line1);
                if processed_line2[0].strip() != cur_stn:
                    continue;
                cur_data_list[6] = float(processed_line2[3]);
    
        # read 3g
        with open(cur_train_3g_datapath) as f_3g:
            for line in f_3g:
                processed_line1 = line.split(',');
                processed_line2 = filter(None, processed_line1);
                if processed_line2[1].strip() != cur_stn:
                    continue;
                stn_num = stn_num + 1;
                if any('DBT' in item for item in processed_line2):
                    dry_bulb_temperature = float(processed_line2[3]);
                    cur_data_list[0] = dry_bulb_temperature;
                if any('DPT' in item for item in processed_line2):
                    dew_point_temperature = float(processed_line2[3]);
                    cur_data_list[1] = dew_point_temperature;
                if any('QFE' in item for item in processed_line2): 
                    atmospheric_pressure_at_aero = float(processed_line2[3]);
                    cur_data_list[2] = atmospheric_pressure_at_aero;
                if any('RH' in item for item in processed_line2): 
                    relative_humidity = float(processed_line2[3]);  
                    cur_data_list[3] = relative_humidity; 
                if any('Wind Dir' in item for item in processed_line2): 
                    wind_dir = float(processed_line2[3]);  
                    cur_data_list[4] = wind_dir; 
                if any('Wind Speed' in item for item in processed_line2): 
                    wind_spd = float(processed_line2[3]); 
                    cur_data_list[5] = wind_spd;
         
        data_list.append(cur_data_list)
    
    return data_list;