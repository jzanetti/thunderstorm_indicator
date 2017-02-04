from mpl_toolkits.basemap import Basemap, cm
import numpy
import matplotlib.pyplot as plt
station_no = ['S06','S100','S102','S104','S106','S107','S108','S109','S111',\
              'S115','S116','S117','S121','S122','S24','S43','S44',\
              'S50','S60','S96'];
station_latlon = [(103.900703,1.352400),\
                  (103.748550,1.417200),\
                  (103.767998,1.189000),\
                  (103.785378,1.443867),\
                   (103.967300,1.416800),\
                   (103.962502,1.313500),\
                   (103.870300,1.279900),\
                   (103.849197,1.376400),\
                   (103.836502,1.310550),\
                   (103.618431,1.293767),\
                   (103.753998,1.281000),\
                   (103.679001,1.256000),\
                   (103.722443,1.372880),\
                   (103.824898,1.417313),\
                   (103.887802,1.339900),\
                   (103.681664,1.345833),\
                   (103.776802,1.333700),\
                   (103.827904,1.250000),\
                   (104.030701,1.317500),\
                   ]

def plot_gauges(station_latlon,station_no,analysis_stn_id):
    analysis_index = station_no.index(analysis_stn_id)
    station_latlon_array = numpy.asarray(station_latlon);
    station_lat_array = station_latlon_array[:,1];
    station_lon_array = station_latlon_array[:,0];
    analysis_lat_array = station_latlon_array[analysis_index,1];
    analysis_lon_array = station_latlon_array[analysis_index,0];
    m = Basemap(llcrnrlat=1.0,urcrnrlat=1.6,llcrnrlon=103.5,urcrnrlon=104.2,resolution='f');
    m.drawcoastlines();
    m.drawstates();
    m.drawcountries();
    x, y = m(station_lon_array,station_lat_array)
    xa, ya = m(analysis_lon_array,analysis_lat_array)
    m.scatter(x,y,3,marker='o',color='k');
    m.scatter(xa,ya,3,marker='o',color='r');
    plt.savefig('test.png');
    plt.close();