from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
import random
import sklearn.linear_model



class FishyFishy():
    def __init__(self, fish_data, future_pred_area, future_sst_daily, sst_daily):

        self.sst_daily = np.load(sst_daily)
        print(self.sst_daily.shape)
        '''
        initialize.
        '''

        self.optimal_temp = dict()
        self.fish_data = dict()
        self.future_pred_area = future_pred_area
        self.fish_types = []
        panic = 0
        for fish, temp, survivable_days, acceptable_range in fish_data:
            self.optimal_temp[fish] = (temp, survivable_days, acceptable_range, panic)
            self.fish_types.append(fish)
        self.future_sst_daily = np.load(future_sst_daily)


        '''
        set the amount of days/5 to observe.
        '''

        self.days = 8000

        
        '''
        generate the coordinates of the docks. 
        '''

        dock_coords = {}
        docks = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        for x in range(1,future_pred_area[0]-1):
            for y in range(1,future_pred_area[1]-1):
                print(np.amin(self.future_sst_daily[10,x-1:x+2,y-1:y+2]))
                
                if self.future_sst_daily[1000,x,y]<(-1000) and np.amax(self.future_sst_daily[10,x-1:x+2,y-1:y+2]) >(-1000):
                    docks[x,y] = 100
                    dock_coords[f'dock_{x}{y}'] = [x,y]
        
        self.docks = docks
        self.dock_coords = dock_coords
        

    def lin_reg(self):


        '''
        run a linear regression for each coordinate and make a new array that holds the prediction extended to 40000 days. 
        '''

        x = np.arange(14245).reshape(-1,1)
        x_future = np.arange(40000).reshape(-1,1)
        
        content = [np.zeros((40000,1,1))]*(future_pred_area[0]*future_pred_area[1])
        future_sst_daily = np.array(content).reshape(40000,future_pred_area[0],future_pred_area[1])
        
        y = self.sst_daily[x,30,30].reshape(-1,1)
        regr = sklearn.linear_model.LinearRegression()
        regr.fit(x,y)
        y_future = regr.predict(x_future).reshape(-1,)
        '''
        for i in range(future_pred_area[0]):
            print(f'{i+1}/80.---')
            for j in range(future_pred_area[1]):
                y = self.sst_daily[x,i,j].reshape(-1,1)
                regr = sklearn.linear_model.LinearRegression()
                regr.fit(x,y)
                y_future = regr.predict(x_future).reshape(-1,)
                

                future_sst_daily[:,i,j] = y_future
        '''

        self.future_sst_daily = future_sst_daily
        #np.save('/Users/makotopowers/Desktop/COMAP_2022_files',future_sst_daily)
        plt.plot(x,y)
        #plt.scatter(x,y)
        plt.plot(x_future,y_future, color='black')
        
        plt.show()



    def heat_map(self):


        '''
        display heatmap of ocean temps.
        '''

        x = np.arange(0,future_pred_area[0])
        y = np.arange(0,future_pred_area[1])
        plt.pcolormesh(x,y,self.future_sst_daily[39999,0:future_pred_area[0],0:future_pred_area[1]].transpose(1,0),vmin=0,vmax=16, cmap='afmhot')
        plt.colorbar()
        plt.show()



    def fish_migration(self):
        
        '''
        initiate arrays for school position and water temp 
        '''

        day = 1
        water_temp = self.future_sst_daily[day*5,0:future_pred_area[0],0:future_pred_area[1]] 
        too_hot_counter = 0
        

        '''
        initiate schools 
        '''

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


        '''
        display initial positions 
        '''

        x = np.arange(0,future_pred_area[0])
        y = np.arange(0,future_pred_area[1])

        initial_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        for school in self.fish_data:
            initial_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1
        overlay_1 = self.future_sst_daily[day,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * initial_pos
        #plt.pcolormesh(x,y,overlay_1.transpose(1,0),vmin=-2, vmax=23, cmap='afmhot')
        #plt.colorbar()
        #plt.show()
        
        
        ''' 
        move fish according to logic for 4*100 days before incrementing temperature.
        '''

        water_temp = self.future_sst_daily[int(day*5),0:future_pred_area[0],0:future_pred_area[1]] 
        for i in range(100):
            print(f'{i+1}/100---')
            mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
            for school in self.fish_data:
                mid_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1

            for school in self.fish_data.keys():
                lat, lon = self.fish_data[school][0], self.fish_data[school][1]  


                '''
                move fish to block with minimum difference between optimal and temp. check position array, if there is already a fish there do not move. 
                if fish gets stuck into a local minimum put it into panic mode. make sure fish do not leave the array. 
                '''

                try:
                    if self.fish_data[school][7] != 0:
                        self.fish_data[school][0] += int(round(random.uniform(-1,1),0))
                        self.fish_data[school][1] += int(round(random.uniform(-1,1),0))
                        self.fish_data[school][7] -= 1
                    else:
                        if np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]) > self.fish_data[school][4]:
                            itemindex = np.where(water_temp[lat-1:lat+2,lon-1:lon+2] == np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]))
                            if mid_pos[min(self.fish_data[school][0] + (itemindex[0][0]-1),78),min(self.fish_data[school][1] + (itemindex[1][0]-1),61)] ==0:
                                self.fish_data[school][0] += (itemindex[0][0]-1)
                                self.fish_data[school][1] += (itemindex[1][0]-1)
                            else:
                                continue
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


        '''
        create an array that holds the positions of all the fish. print. this shows where the fish would start. 
        '''

        mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        for school in self.fish_data:
            mid_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1

        
        overlay_2 = self.future_sst_daily[day,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * mid_pos
        plt.pcolormesh(x,y,overlay_2.transpose(1,0),vmin=-2, vmax=23)
        plt.colorbar()
        plt.show()


        '''
        initiate the out of range dictionary and map that represents the out of range matrix
        '''

        out_of_range = {}
        out_of_range_map = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))  


        '''
        loop through the days. set the water temp to the according water temp by copying the matrix. move fish according to logic. 
        check the condition of the fish; if it is too hot they die. move fish 3 times for each time increment. 
        check if the fish are out of range, for each dock. update the out of range dock data. 
        '''

        for day in range(self.days-1):
            print(f'Day: {day+1}/{self.days} :'+'*'*int(((day+1)/self.days)*20//1))
            water_temp = self.future_sst_daily[int(day*5),0:future_pred_area[0],0:future_pred_area[1]] 

            mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
            for school in self.fish_data:
                mid_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1
            
            killed = []

            for school in self.fish_data.keys():
                lat, lon = self.fish_data[school][0], self.fish_data[school][1]  

                try:
                    if self.fish_data[school][7] != 0:
                        self.fish_data[school][0] += int(round(random.uniform(-1,1),0))
                        self.fish_data[school][1] += int(round(random.uniform(-1,1),0))
                        self.fish_data[school][7] -= 1
                    else:
                        if np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]) > self.fish_data[school][4]:
                            itemindex = np.where(water_temp[lat-1:lat+2,lon-1:lon+2] == np.amin(water_temp[lat-1:lat+2,lon-1:lon+2]))
                            if mid_pos[min(self.fish_data[school][0] + (itemindex[0][0]-1),78),min(self.fish_data[school][1] + (itemindex[1][0]-1),61)] ==0:
                                self.fish_data[school][0] += (itemindex[0][0]-1)
                                self.fish_data[school][1] += (itemindex[1][0]-1)
                            else:
                                continue
                        else:
                            diff = np.abs(water_temp[lat-1:lat+2,lon-1:lon+2] - self.fish_data[school][4])
                            itemindex = np.where(diff == np.amin(diff))
                            self.fish_data[school][0] += (itemindex[0][0]-1)
                            self.fish_data[school][1] += (itemindex[1][0]-1)
                        if (itemindex[0][0]-1) == 0 and (itemindex[1][0]-1) == 0 and self.fish_data[school][7] == 0:
                            self.fish_data[school][7] = 10 
                    
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

                
                if water_temp[self.fish_data[school][0], self.fish_data[school][1]] > (self.fish_data[school][4] + self.fish_data[school][6]):
                    self.fish_data[school][2] += 1
                else:
                    self.fish_data[school][2] = 0
                if self.fish_data[school][2] > self.fish_data[school][5]:
                    killed.append(school)
            if killed:
                for death in killed:
                    self.fish_data.pop(death)
                killed = []


            
            dock_range = 3
            mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
            for school in self.fish_data:
                mid_pos[self.fish_data[school][0],self.fish_data[school][1]] =1
            for dock in self.dock_coords:
                x,y = self.dock_coords[dock]
                if np.sum(mid_pos[max(x-dock_range,0):min(x+dock_range+1,future_pred_area[0]),max(y -dock_range,0):min(y+dock_range+1, future_pred_area[1])])<4:
                    try:
                        out_of_range[dock]=min(out_of_range[dock], day)
                    except KeyError:
                        out_of_range[dock] = day
                else:
                    out_of_range[dock] = day


        '''
        display the plot for how long it takes fish to move out of range.
        '''

        for dock in out_of_range:
            x,y =self.dock_coords[dock]
            out_of_range_map[x,y] = out_of_range[dock]
        
        x = np.arange(0,future_pred_area[0])
        y = np.arange(0,future_pred_area[1])
        #plt.pcolormesh(x,y,out_of_range_map.transpose(1,0),vmin=0,vmax=2000, cmap='afmhot')
        #plt.colorbar()
        #plt.show()
            

        '''
        display the final position.
        '''

        final_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        try:
            for school in self.fish_data:
                if self.fish_data[school][3] == 'herring':
                    final_pos[self.fish_data[school][0],self.fish_data[school][1]] +=1
                if self.fish_data[school][3] == 'mackerel':
                    final_pos[self.fish_data[school][0],self.fish_data[school][1]] +=10
        except IndexError:
            print(self.fish_data[school][0],self.fish_data[school][1])

        overlay_2 = self.future_sst_daily[day,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * final_pos
        #plt.pcolormesh(x,y,overlay_2.transpose(1,0), vmin=-2, vmax=23, cmap='afmhot')
        #plt.show()


   
'''
files.
'''

#ocean_data = 'sst.wkmean.1990-present.nc'
#future_sst = 'future_sst.npy'
daily_sst = 'daily_ssts.npy'

future_sst_daily = 'future_sst_daily.npy'


'''
fish data.
'''

fish_data = [('mackerel', 9, 7, 3)]
#fish_data = [('herring', 16, 7, 3)]
#('mackerel', 9, 4, 3)
#('herring', 16, 3, 2)


'''
future area of prediction. 
'''

future_pred_area = [80,64]


'''
instantiate.
'''

fish = FishyFishy(fish_data, future_pred_area, future_sst_daily, daily_sst)


fish.fish_migration()
#fish.lin_reg()

#fish.heat_map()

