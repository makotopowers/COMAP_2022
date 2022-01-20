from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
import seaborn
import xarray as xr
import sklearn


class FishyFishy():
    def __init__(self, temp_data):
        self.temp_data = temp_data
        self.optimal_temp = dict()
        self.fish_data = dict()

        #dummy variables for testing
        self.days = range(10)


    def lin_reg(self):
        #interpolate between heat data so there is new daily data
        self.interval = np.ndarray() #needs to be (day, heat, x, y)
        pass


    def fish_temps(self, fish_temp_data):
        self.fish_types = []
        for fish, temp, survivable_days, acceptable_range in fish_temp_data:
            self.optimal_temp[fish] = (temp, survivable_days, acceptable_range)
            self.fish_types.append(fish)


    def heat_map(self, region=None):
        #heatmap of ocean temp
        ds = xr.open_dataset(self.temp_data)
        df = ds.to_dataframe()
        print(df.head(10))


        fp=self.temp_data
        nc = netCDF4.Dataset(fp)
        print(type(nc))
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        time = nc.variables['time'][:]
        sst = nc.variables['sst'][:] # 2 meter temperature
        time_bnds = nc.variables['time_bnds'][:] # mean sea level pressure
        self.sst = nc.variables['sst'][:]
        
        #print(f'LAT_SHAPE: {lat.shape}')
        #print(f'LON_SHAPE: {lon.shape}')
        print(f'SST_shape: {sst.shape}')
        #print(f'SST: {sst}')
        print(f'{sst[1,50:65,-15:-1]}')
        print(f'{sst[1,50:65,-15:-1].shape}')
        #print(lat)

        #seaborn.heatmap(self.sst[1:2,0:-15,50:65])
        #plt.show()

    def fish_migration(self, depth):
        
        #initiate arrays for school position and water temp 
        fish_pos = np.zeros((15,15))
        water_temp = self.sst[depth,50:65,-15:0] # self.interpolated once lin reg
        print(f'water_temp: {water_temp}')
        too_hot_counter = 0
        
        #initiate schools 
        for i in range(1,16): 

            #start fish at random positions
            ax, ay = np.random.random(2)*15//1 
            if ax == 0:
                ax = ax+1
            elif ax ==14:
                ax = ax-1
            else: 
                ax = ax
            if ay == 0:
                ay = ay+1
            elif ay ==14:
                ay = ay-1
            else: 
                ay = ay

            x,y = int(ax), int(ay)
        
            print(f'X:{x}, Y:{y}')

            random_type = np.random.choice(self.fish_types)

            print(f'RAND_TYPE: {random_type}')
            self.fish_data[f'school_{i}'] = [x,y,too_hot_counter, random_type, self.optimal_temp[random_type][0], \
                self.optimal_temp[random_type][1], self.optimal_temp[random_type][2]]
            
        print(f'FISH_DATA: {self.fish_data}')

        to_plot = []
        #loop through time interval 
        for day in self.days:
            print(f'Day: {day}')
            #for each day, change the position of each fish school 
            for school in self.fish_data.keys():

                lat, lon = self.fish_data[school][0], self.fish_data[school][1]  

                #fish movement logic
                print(lat, lon)
                #abs(self.optimal_temp[self.fish_data[school][2]] -

                #move fish to block with minimum difference between optimal and temp
                try:
                    print(water_temp[lat-1:lat+2,lon-1:lon+2])
                    print(np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]))
                
                    items = (water_temp[lat-1:lat+2,lon-1:lon+2] == np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]))
                    print(items)
                    itemindex = np.where(items)
                    self.fish_data[school][0] += (itemindex[0][0]-1)
                    self.fish_data[school][1] += (itemindex[1][0]-1)
                except ValueError:
                    print('array has size 0.')

                if water_temp[self.fish_data[school][0], self.fish_data[school][1]] > (self.fish_data[school][4] + self.fish_data[school][6]):
                    self.fish_data[school][2] += 1
                else:
                    self.fish_data[school][2] = 0

                if self.fish_data[school][2] > self.fish_data[school][5]:
                    killed = self.fish_data.pop(school)
                    print(f'{school} was killed.')



            fishes = []
            for school in self.fish_data.keys():
                fishes.append(self.fish_data[school][0:2])
                print(self.fish_data[school][0:2])
            print(fishes)
            fishes = [list(x) for x in zip(*fishes)]
            print(fishes)  
            
            to_plot.append(fishes)
            
        #plt.title(f'Day: {i}')
        #x = plt.scatter(fishies[0], fishies[1])
        #plt.show()
        
        print(f'to_plot: {to_plot}')
        fig, ax = plt.subplots(3,3 , figsize=(20, 20), sharex = True, sharey = True)
        for i in range(3):
            for j in range(3):
                if 4*i + j >9:
                    pass
                else:
                    
                    ax[i,j].scatter(to_plot[4*i+j][0],to_plot[2*i+j][1])
                    #ax[i,j].set_xlabel("lat")
                    #ax[i,j].set_ylabel("lon")
                    #ax[i,j].xlim(0,15)
                    #ax[i,j].ylim(0,15)
                    ax[i,j].set_title(f'Day: {3*i+j+1}')

        plt.show()
        #for day in self.interval:
            #water_temp = self.sst[depth,:15,:15] # self.interpolated[depth,:15,:15,day]
            
    def run(self, fish_temp_data, depth):
        self.fish_temps(fish_temp_data)
        self.heat_map()
        self.fish_migration(depth)

        

   
path = '/Users/makotopowers/Desktop/sst.wkmean.1990-present.nc'


fish_temp_data = [('herring', 4.6, 3, 2), ('mackerel', 5, 4, 3)]
sol = FishyFishy(path)

#sol.run(fish_temp_data, 1)

sol.heat_map()