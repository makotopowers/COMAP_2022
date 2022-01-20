from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
import seaborn
import xarray as xr
import sklearn.linear_model
import time


class FishyFishy():
    def __init__(self, ocean_data, fish_data):
        #self.temp_data = temp_data
        self.optimal_temp = dict()
        self.fish_data = dict()
        self.weeks = range(1671)
        #fp=self.temp_data
        nc = netCDF4.Dataset(ocean_data)
        self.sst = nc.variables['sst'][:] # shape = (1672,180,360)

        self.fish_types = []
        for fish, temp, survivable_days, acceptable_range in fish_data:
            self.optimal_temp[fish] = (temp, survivable_days, acceptable_range)
            self.fish_types.append(fish)


    def lin_reg(self):
        x = np.arange(1671).reshape(-1,1)
        y = self.sst[x,50,50].reshape(-1,1)
        regr = sklearn.linear_model.LinearRegression()
        regr.fit(x,y)

        future_sst = np.zeros((3000,180,360))

        x_future = np.arange(3000).reshape(-1,1)
        y_future = regr.predict(x_future)
        plt.plot(x,y,color="black")

        #plt.scatter(x_future, y_future, color="black")
        plt.plot(x_future, y_future, color="blue", linewidth=3)       
        plt.show()
        #print(y)
        #for i in range(180):
            #for j in range(360):

                


    def heat_map(self):
        x = np.arange(0,100)
        y = np.arange(0,100)
        plt.pcolormesh(x,y,self.sst[500,0:100,0:100])
        plt.colorbar()
        plt.show()



    def fish_migration(self):
        
        #initiate arrays for school position and water temp 
        week = 1
        water_temp = self.sst[week,0:100,0:100] 
        too_hot_counter = 0
        
        size = 99 #size of region

        #initiate schools 
        for i in range(1,200): #how many schools
            ax, ay = np.random.random(2)*size//1 
            if ax == 0:
                ax = ax+1
            elif ax ==size:
                ax = ax-1
            if ay == 0:
                ay = ay+1
            elif ay ==size:
                ay = ay-1
 
            x,y = int(ax), int(ay)

            random_type = np.random.choice(self.fish_types)

            self.fish_data[f'school_{i}'] = [x,y,too_hot_counter, random_type, self.optimal_temp[random_type][0], \
                self.optimal_temp[random_type][1], self.optimal_temp[random_type][2]]

            
        print(f'FISH_DATA: {self.fish_data}')

        #done initiating

        #display initial positions

        x = np.arange(0,100)
        y = np.arange(0,100)

        initial_pos = np.zeros((99,99))
        for school in self.fish_data:
            initial_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1


        overlay_1 = self.sst[500,0:99,0:99] + 20 * initial_pos
        #plt.pcolormesh(x,y,overlay_1)
        #plt.colorbar()
        #plt.show()



        
        #loop through time interval 
        for week in self.weeks:
            #print(f'Week: {week}')

            #change the position of each school 
            killed = []

            for school in self.fish_data.keys():
                lat, lon = self.fish_data[school][0], self.fish_data[school][1]  

                #move fish to block with minimum difference between optimal and temp
                try:
                    itemindex = np.where(water_temp[lat-1:lat+2,lon-1:lon+2] == np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]))
                    self.fish_data[school][0] += (itemindex[0][0]-1)
                    self.fish_data[school][1] += (itemindex[1][0]-1)
                except ValueError:
                    print('array has size 0.')

                #check condition of school

                if water_temp[self.fish_data[school][0], self.fish_data[school][1]] > (self.fish_data[school][4] + self.fish_data[school][6]):
                    self.fish_data[school][2] += 1
                else:
                    self.fish_data[school][2] = 0

                #if self.fish_data[school][2] > self.fish_data[school][5]:
                    #killed.append(school)
                    #print(f'{school} was killed.')

            if killed:
                for death in killed:
                    self.fish_data.pop(death)
                killed = []

        #display final position
        final_pos = np.zeros((100,100))
        for school in self.fish_data:
            final_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1

        overlay_2 = self.sst[500,0:100,0:100] + 20 * final_pos
        plt.pcolormesh(x,y,overlay_2)

        plt.show()
        #for day in self.interval:
            #water_temp = self.sst[depth,:15,:15] # self.interpolated[depth,:15,:15,day]
            
    def run(self, fish_temp_data, depth):
        self.fish_migration(depth)

        

   
ocean_data = '/Users/makotopowers/Desktop/sst.wkmean.1990-present.nc'
fish_data = [('herring', 4.6, 3, 2), ('mackerel', 5, 4, 3)]

fish = FishyFishy(ocean_data, fish_data)

fish.fish_migration()

#fish.heat_map()

#fish.lin_reg()