from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
import random
import sklearn.linear_model



class FishyFishy():
    def __init__(self, ocean_data, fish_data, future_pred_area, future_sst):
        #self.temp_data = temp_data
        self.optimal_temp = dict()
        self.fish_data = dict()
        self.weeks = range(600)
        #fp=self.temp_data
        nc = netCDF4.Dataset(ocean_data)
        self.sst = nc.variables['sst'][:].transpose(0,2,1) # shape = (1672,180,360)


        self.future_pred_area = future_pred_area
        self.fish_types = []
        panic = 0
        for fish, temp, survivable_days, acceptable_range in fish_data:
            self.optimal_temp[fish] = (temp, survivable_days, acceptable_range, panic)
            self.fish_types.append(fish)
        self.future_sst = np.load(future_sst).transpose(0,2,1)


    def lin_reg(self):
        x = np.arange(1671).reshape(-1,1)
        content = [np.zeros((3000,1,1))]*(future_pred_area[0]*future_pred_area[1])
        y = self.sst[x,45,300].reshape(-1,1)

        regr = sklearn.linear_model.LinearRegression()
        regr.fit(x,y)
        x_future = np.arange(3000).reshape(-1,1)
        y_future = regr.predict(x_future).reshape(-1,)
        print(regr.predict(np.arange(10,15).reshape(-1,1)))
        print(regr.predict(np.arange(2986,2988).reshape(-1,1)))
        plt.plot(x,y)
        plt.plot(x_future,y_future, color='black')
        plt.show()

        #Code has already been run. Get array from self.future_sst
        '''
        content = [np.zeros((3000,1,1))]*(future_pred_area[0]*future_pred_area[1])
        future_sst = np.array(content).reshape(3000,future_pred_area[0],future_pred_area[1])
        
        for i in range(future_pred_area[0]):
            for j in range(future_pred_area[1]):
                y = self.sst[x,i,j].reshape(-1,1)
                regr = sklearn.linear_model.LinearRegression()
                regr.fit(x,y)
                y_future = regr.predict(x_future).reshape(-1,)
                

                future_sst[:,i,j] = y_future

        self.future_sst = future_sst
        np.save('/Users/makotopowers/Desktop', future_sst)
        '''



    def heat_map(self):
        x = np.arange(0,future_pred_area[1])
        y = np.arange(0,future_pred_area[0])
        plt.pcolormesh(x,y,self.future_sst[100,0:future_pred_area[0],0:future_pred_area[1]])
        plt.colorbar()
        plt.show()



    def fish_migration(self):
        
        #initiate arrays for school position and water temp 
        week = 1
        water_temp = self.future_sst[week*5,0:future_pred_area[0],0:future_pred_area[1]] 
        too_hot_counter = 0
        
       

        #initiate schools 
        for i in range(1,1000): #how many schools
            ax = np.random.random(1)*(future_pred_area[0]-1)//1 
            ay = np.random.random(1)*(future_pred_area[1]-1)//1 
            if ax == 0:
                ax = ax+1
            elif ax ==future_pred_area[0]-1:
                ax = ax-1
            if ay == 0:
                ay = ay+1
            elif ay ==future_pred_area[1]-1:
                ay = ay-1
 
            x,y = int(ax), int(ay)

            random_type = np.random.choice(self.fish_types)

            self.fish_data[f'school_{i}'] = [x,y,too_hot_counter, random_type, self.optimal_temp[random_type][0], \
                self.optimal_temp[random_type][1], self.optimal_temp[random_type][2], self.optimal_temp[random_type][3]]

            
        print(f'FISH_DATA: {self.fish_data}')

        #done initiating

        #display initial positions

        x = np.arange(0,future_pred_area[0])
        y = np.arange(0,future_pred_area[1])

        initial_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        for school in self.fish_data:
            initial_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1

        '''
        overlay_1 = self.future_sst[week,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * initial_pos
        plt.pcolormesh(x,y,overlay_1.transpose(1,0))
        #plt.colorbar()
        plt.show()
        '''


        
        #loop through time interval 
        for week in self.weeks:
            #print(f'Week: {week}')
            water_temp = self.future_sst[int(week*5),0:future_pred_area[0],0:future_pred_area[1]] 
            #change the position of each school 
            killed = []

            for school in self.fish_data.keys():
                lat, lon = self.fish_data[school][0], self.fish_data[school][1]  

                #move fish to block with minimum difference between optimal and temp
                try:
                    if self.fish_data[school][7] != 0:
                        self.fish_data[school][0] += int(round(random.uniform(-1,1),0))
                        self.fish_data[school][1] += int(round(random.uniform(-1,1),0))
                        self.fish_data[school][7] -= 1
                    else:
                        if np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]) > self.fish_data[school][4]:
                            itemindex = np.where(water_temp[lat-1:lat+2,lon-1:lon+2] == np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]))
                            self.fish_data[school][0] += (itemindex[0][0]-1)
                            self.fish_data[school][1] += (itemindex[1][0]-1)
                        else:
                            diff = np.abs(water_temp[lat-1:lat+2,lon-1:lon+2] - self.fish_data[school][4])
                            itemindex = np.where(diff == np.amin(diff))
                            self.fish_data[school][0] += (itemindex[0][0]-1)
                            self.fish_data[school][1] += (itemindex[1][0]-1)
                        if (itemindex[0][0]-1) == 0 and (itemindex[1][0]-1) == 0 and self.fish_data[school][7] == 0:
                            self.fish_data[school][7] = 10 #have the fish panic for 10 days
                    
                    if self.fish_data[school][0] > future_pred_area[0]-2:
                        self.fish_data[school][0] = future_pred_area[0]-2
                    if self.fish_data[school][0] < 1:
                        self.fish_data[school][0] += 1
                    if self.fish_data[school][1] > future_pred_area[1]-2:
                        self.fish_data[school][1] = future_pred_area[1]-2
                    if self.fish_data[school][1] < 1:
                        self.fish_data[school][1] += 1


                except ValueError:
                    continue

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
        final_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        try:
            for school in self.fish_data:
                final_pos[self.fish_data[school][0],self.fish_data[school][1]] =1
        except IndexError:
            print(self.fish_data[school][0],self.fish_data[school][1])

        overlay_2 = self.future_sst[week,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * final_pos
        plt.pcolormesh(x,y,overlay_2.transpose(1,0))
        plt.show()
        



    def boat_movement(self):
        pass

        

   
ocean_data = '/Users/makotopowers/Desktop/sst.wkmean.1990-present.nc'
fish_data = [('herring', 4.6, 3, 2), ('mackerel', 5, 4, 3)]
future_pred_area = [360,180]
future_sst = '/Users/makotopowers/Desktop/future_sst.npy'

fish = FishyFishy(ocean_data, fish_data, future_pred_area, future_sst)

fish.fish_migration()


#fish.lin_reg()
#fish.heat_map()
