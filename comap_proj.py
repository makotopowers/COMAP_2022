from re import S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
import random
import sklearn.linear_model



class FishyFishy():
    def __init__(self, fish_data, future_pred_area, daily_sst, future_sst_daily):
        self.optimal_temp = dict()
        self.fish_data = dict()
        self.weeks = 8000
        self.future_pred_area = future_pred_area
        self.fish_types = []
        panic = 0
        for fish, temp, survivable_days, acceptable_range in fish_data:
            self.optimal_temp[fish] = (temp, survivable_days, acceptable_range, panic)
            self.fish_types.append(fish)
        #self.sst_daily = np.load(daily_sst).transpose(0,2,1)
        self.future_sst_daily = np.load(future_sst_daily)

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
        #print(self.dock_coords)
        #print(docks)
        #x = np.arange(0,80)
        #y = np.arange(0,64)

        #plt.pcolormesh(x,y,docks.transpose(1,0))
        #plt.show()
        

    def lin_reg(self):
        x = np.arange(14245).reshape(-1,1)
        #content = [np.zeros((40000,1,1))]*(future_pred_area[0]*future_pred_area[1])
        #y = self.sst_daily[x,80,64].reshape(-1,1)

        #regr = sklearn.linear_model.LinearRegression()
        #regr.fit(x,y)
        x_future = np.arange(40000).reshape(-1,1)
        #y_future = regr.predict(x_future).reshape(-1,)
        #print(regr.predict(np.arange(10,15).reshape(-1,1)))
        #print(regr.predict(np.arange(2986,2988).reshape(-1,1)))
        #plt.plot(x,y)
        #plt.plot(x_future,y_future, color='black')
        #plt.show()


        #Code has already been run. Get array from self.future_sst
        
        content = [np.zeros((40000,1,1))]*(future_pred_area[0]*future_pred_area[1])
        future_sst_daily = np.array(content).reshape(40000,future_pred_area[0],future_pred_area[1])
        
        for i in range(future_pred_area[0]):
            print(f'{i+1}/80.---')
            for j in range(future_pred_area[1]):
                y = self.sst_daily[x,i,j].reshape(-1,1)
                regr = sklearn.linear_model.LinearRegression()
                regr.fit(x,y)
                y_future = regr.predict(x_future).reshape(-1,)
                

                future_sst_daily[:,i,j] = y_future

        self.future_sst_daily = future_sst_daily
        #np.save('/Users/makotopowers/Desktop/COMAP_2022_files',future_sst_daily)
        print(self.future_sst_daily.shape)
        print(self.future_sst_daily[0,25,20])
        print(self.future_sst_daily[10000,25,20])
        print(self.future_sst_daily[20000,25,20])
        print(self.future_sst_daily[38000,25,20])
        plt.plot(x,y)
        plt.plot(x_future,y_future, color='black')
        plt.show()



    def heat_map(self):
        x = np.arange(0,future_pred_area[0])
        y = np.arange(0,future_pred_area[1])
        plt.pcolormesh(x,y,self.future_sst_daily[0,0:future_pred_area[0],0:future_pred_area[1]].transpose(1,0),vmin=0)
        plt.colorbar()
        plt.show()



    def fish_migration(self):
        
        #initiate arrays for school position and water temp 
        week = 1
        water_temp = self.future_sst_daily[week*5,0:future_pred_area[0],0:future_pred_area[1]] 
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

        print(initial_pos.shape)
        print(self.future_sst_daily.shape)
        
        overlay_1 = self.future_sst_daily[week,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * initial_pos
        #plt.pcolormesh(x,y,overlay_1.transpose(1,0),vmin=-2, vmax=23)
        #plt.colorbar()
        #plt.show()
        
        water_temp = self.future_sst_daily[int(week*5),0:future_pred_area[0],0:future_pred_area[1]] 
            #change the position of each school 

        for i in range(100):
            print(f'{i+1}/100---')
            mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
            for school in self.fish_data:
                mid_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1

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

        mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        for school in self.fish_data:
            mid_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1

        
        overlay_2 = self.future_sst_daily[week,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * mid_pos
        #plt.pcolormesh(x,y,overlay_2.transpose(1,0),vmin=-2, vmax=23)
        #plt.colorbar()
        #plt.show()


        out_of_range = {}
        out_of_range_map = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))  


        #loop through time interval 
        for week in range(self.weeks-1):
            print(f'Week: {week+1}/{self.weeks} :'+'*'*int(((week+1)/self.weeks)*20//1))
            water_temp = self.future_sst_daily[int(week*5),0:future_pred_area[0],0:future_pred_area[1]] 
            #change the position of each school 

            mid_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
            for school in self.fish_data:
                mid_pos[self.fish_data[school][0],self.fish_data[school][1]] = 1
            
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

                #check condition of school
                
                if water_temp[self.fish_data[school][0], self.fish_data[school][1]] > (self.fish_data[school][4] + self.fish_data[school][6]):
                    self.fish_data[school][2] += 1
                else:
                    self.fish_data[school][2] = 0

                if self.fish_data[school][2] > self.fish_data[school][5]:
                    killed.append(school)
                    #print(f'{school} was killed.')

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
                        out_of_range[dock]=min(out_of_range[dock], week)
                    except KeyError:
                        out_of_range[dock] = week
                else:
                    out_of_range[dock] = week


        #print(out_of_range)
        for dock in out_of_range:
            
            
            x,y =self.dock_coords[dock]
            out_of_range_map[x,y] = out_of_range[dock]
                        
        
        print(out_of_range)
        
        x = np.arange(0,future_pred_area[0])
        y = np.arange(0,future_pred_area[1])
        
        plt.pcolormesh(x,y,out_of_range_map.transpose(1,0),vmin=0,vmax=2000)
        plt.colorbar()
        plt.show()
            

        


        #display final position
        final_pos = np.zeros((future_pred_area[0]-1,future_pred_area[1]-1))
        try:
            for school in self.fish_data:
                if self.fish_data[school][3] == 'herring':
                    final_pos[self.fish_data[school][0],self.fish_data[school][1]] +=1
                if self.fish_data[school][3] == 'mackerel':
                    final_pos[self.fish_data[school][0],self.fish_data[school][1]] +=10
        except IndexError:
            print(self.fish_data[school][0],self.fish_data[school][1])

        overlay_2 = self.future_sst_daily[week,0:future_pred_area[0]-1,0:future_pred_area[1]-1] + 20 * final_pos
        #plt.pcolormesh(x,y,overlay_2.transpose(1,0), vmin=-2, vmax=23)
        #plt.show()
        



    
            



        

   
#ocean_data = 'sst.wkmean.1990-present.nc'
fish_data = [('mackerel', 9, 7, 3)]
#fish_data = [('herring', 16, 7, 3)]

#('mackerel', 9, 4, 3)
#('herring', 16, 3, 2)
future_pred_area = [80,64]
future_sst = 'future_sst.npy'
daily_sst = 'daily_ssts.npy'
future_sst_daily = 'future_sst_daily.npy'

fish = FishyFishy(fish_data, future_pred_area, daily_sst, future_sst_daily)



fish.fish_migration()
#fish.lin_reg()
#fish.heat_map()
