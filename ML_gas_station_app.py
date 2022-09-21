from cProfile import label
from tkinter import *
from tkinter import simpledialog
from tkinter import ttk
from tkinter import Canvas
from tkinter import Frame
from tkinter import Text
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import pickle
import glob
from os.path import exists,isfile,isdir
from os import remove,rmdir,system
from functools import partial
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import torch
from tqdm import tqdm
from PIL import Image,ImageTk


class extract_tensor(torch.nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor


class LSTM_model():
    def __init__(self,train_data,gas_station_name,column):
        self.batch_size = 24
        self.train_data = train_data
        self.gas_station_name = gas_station_name
        self.column = column
        self.diff_data = self.data_diff(train_data[0])
        self.diff_data = np.array(self.diff_data).reshape(-1,1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.diff_scaled_data = self.scaler.fit_transform(self.diff_data)
        self.train_X,self.train_Y = self.prepare_data()
        self.train_Y = self.train_Y.squeeze()

        device = torch.device('cpu')
        layers = []
        layers.append(torch.nn.LSTM(input_size = 24,hidden_size=80,num_layers=2,dropout=0.1,batch_first=False))
        layers.append(extract_tensor())
        layers.append(torch.nn.Linear(80,50))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(50,1))
        self.model = torch.nn.Sequential(*layers).to(device)
        
        self.train()
        self.save_model()

    def prepare_data(self, look_back=24):
        X,Y = [],[]
        for i in range(len(self.diff_scaled_data)-look_back):
            a = self.diff_scaled_data[i:(i+look_back)]
            X.append(a)
            Y.append(self.diff_scaled_data[i + look_back])
        return np.array(X),np.array(Y)

    def data_diff(self,prev):
        new_data = [self.train_data[0]-prev]
        for i in range(len(self.train_data)-1):
            new_data.append(self.train_data[i+1]-self.train_data[i])
        return new_data

    def summary(self):
        print(self.model.summary())

    def train(self):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        batches_num = int(self.train_X.shape[0]/self.batch_size)

        torch.from_numpy(self.train_X).float()
        torch.from_numpy(self.train_Y).float()

        for _ in tqdm(range(60),desc = 'Train'):
            for i in range(batches_num):
                inputs = torch.from_numpy(self.train_X[self.batch_size*i:self.batch_size*(i+1)]).float()
                inputs = torch.transpose(inputs,0,2)
                inputs = torch.transpose(inputs,1,2)
                
                targets = torch.from_numpy(self.train_Y[self.batch_size*i:self.batch_size*(i+1)]).float()
                targets = torch.unsqueeze(torch.unsqueeze(targets,1),2)
                targets = torch.transpose(targets,0,1)

                optimizer.zero_grad()
                yhat = self.model(inputs)
                loss = criterion(yhat, targets)
                loss.backward()
                optimizer.step()

    def save_model(self):
        torch.save(self.model,r'models\{gas_station_name}_{column}'.format(gas_station_name = self.gas_station_name,column = self.column))
        outfile = open('models\{gas_station_name}_{column}.sav'.format(gas_station_name = self.gas_station_name,column = self.column), 'wb')
        pickle.dump([self.scaler,self.train_data[len(self.train_data)-self.batch_size-1:]], outfile)

class LSTM_model_update():
    def __init__(self,data,gas_station_name,column):
        self.data = data
        self.gas_station_name = gas_station_name
        self.column = column
        self.batch_size = 24
        self.load_model()
        self.prev += self.data

        self.diff_data = self.data_diff(0)
        self.diff_data.pop(0)
        self.diff_data = np.array(self.diff_data).reshape(-1,1)
        self.diff_scaled_data = self.scaler.transform(self.diff_data)
        self.X,self.Y = self.prepare_data()

        self.X = torch.from_numpy(self.X).float()
        self.X = torch.transpose(self.X,0,2)
        self.X = torch.transpose(self.X,1,2)
        self.Y = torch.from_numpy(self.Y).float()
        diff_preds = self.model(self.X)

        self.train()
        diff_preds = self.scaler.inverse_transform(diff_preds.squeeze().detach().numpy().reshape(-1,1))
        
        self.preds = []
        
        for i in range(len(diff_preds)):
            self.preds.append(self.prev[i]+diff_preds[i])
        self.save_model()

    def load_model(self):
        self.model = torch.load(r'models\{gas_station_name}_{column}'.format(gas_station_name = self.gas_station_name,column = self.column))
        self.scaler,self.prev = pickle.load(open('models\{gas_station_name}_{column}.sav'.format(gas_station_name = self.gas_station_name,column = self.column),"rb"))

    def prepare_data(self, look_back=24):
        X,Y = [],[]
        for i in range(len(self.diff_scaled_data)-look_back):
            a = self.diff_scaled_data[i:(i+look_back)]
            X.append(a)
            Y.append(self.diff_scaled_data[i + look_back])
        return np.array(X),np.array(Y)

    def data_diff(self,prev):
        new_data = [self.prev[0]-prev]
        for i in range(len(self.prev)-1):
            new_data.append(self.prev[i+1]-self.prev[i])
        return new_data
        
    def get_preds(self):
        return self.preds

    def train(self):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in tqdm(range(1),desc = 'Test'):
            inputs = self.X
            targets = self.Y
            targets = torch.unsqueeze(targets,1)
            targets = torch.transpose(targets,0,1)
            optimizer.zero_grad()
            yhat = self.model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

    def save_model(self):
        torch.save(self.model,r'models\{gas_station_name}_{column}'.format(gas_station_name = self.gas_station_name,column = self.column))
        outfile = open('models\{gas_station_name}_{column}.sav'.format(gas_station_name = self.gas_station_name,column = self.column), 'wb')
        pickle.dump([self.scaler,self.prev[len(self.prev)-self.batch_size-1:]], outfile)

        outfile2 = open('models\{gas_station_name}_{column}_preds.sav'.format(gas_station_name = self.gas_station_name,column = self.column), 'wb')
        pickle.dump([self.preds,self.data], outfile2)

def close_window():
    main_window.destroy()
    exit()


def command1():
    loc = simpledialog.askstring(title="Import new gas stations", prompt="File path:")
    df = pd.read_excel(loc, sheet_name=None)

    new_gas_stations = [c for c in list(df.keys()) if c not in gas_stations_list]
    if len(new_gas_stations) == 0:
        return

    progress_window = Toplevel()
    progress_window.title('Training the models...')
    progress_window.resizable(False,False)
    progress_bars = []
    percentage_labels = []
    for i in range(len(new_gas_stations)):
        Label(progress_window,text = new_gas_stations[i]).grid(row=i,column=0)
        progress_bars.append(ttk.Progressbar(progress_window,orient=HORIZONTAL,length=300,mode='determinate'))
        progress_bars[i].grid(row=i,column=1)
        percentage_labels.append(Label(progress_window,text='0%'))
        percentage_labels[i].grid(row=i,column=2)
        progress_window.update()
    
    counter = -1
    for key in df.keys():
        if key in gas_stations_list:
            continue
        counter+=1    
        gas_stations_list.append(key)
        import_options_menu.add_command(label=key,command=partial(command3,key))
        import_options_menu.entryconfig(key, state="disabled")
        delete_menu.add_command(label=key,command=partial(command4,key))

        df[key].to_csv('%s.csv' %key)

        temp_df = pd.read_csv('%s.csv' %key)
        temp_df  = temp_df.iloc[: , :9]

        #ADDING MODELS FOR EVERY COLUMN
        data1 = list(temp_df.iloc[:,3].values)
        del data1[0:len(data1):25]
        LSTM_model(data1,key,'field_volume')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()
        data2 = list(temp_df.iloc[:,5].values)
        del data2[0:len(data2):25]
        LSTM_model(data2,key,'hcv')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()
        data3 = list(temp_df.iloc[:,7].values)
        del data3[0:len(data3):25]
        LSTM_model(data3,key,'pressure')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()
        data4 = list(temp_df.iloc[:,8].values)
        del data4[0:len(data4):25]
        LSTM_model(data4,key,'temperature')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()

    progress_window.destroy()
    for file in glob.glob("*.csv"):
        remove(file)

def command2():
    loc = simpledialog.askstring(title="Update existing gas stations", prompt="File path:")
    df = pd.read_excel(loc, sheet_name=None)

    gas_stations_found = [c for c in list(df.keys()) if c in gas_stations_list]
    if len(gas_stations_found) == 0:
        return

    progress_window = Toplevel()
    progress_window.title('Generating predictions...')
    progress_window.resizable(False,False)
    progress_bars = []
    percentage_labels = []
    for i in range(len(gas_stations_found)):
        Label(progress_window,text = gas_stations_found[i]).grid(row=i,column=0)
        progress_bars.append(ttk.Progressbar(progress_window,orient=HORIZONTAL,length=300,mode='determinate'))
        progress_bars[i].grid(row=i,column=1)
        percentage_labels.append(Label(progress_window,text='0%'))
        percentage_labels[i].grid(row=i,column=2)
        progress_window.update()

    counter =-1
    for key in df.keys():
        if key not in gas_stations_list:
            continue
        counter += 1
        import_options_menu.entryconfig(key, state="active")
        df[key].to_csv('%s_latest.csv' %key)

        temp_df = pd.read_csv('%s_latest.csv' %key)
        if temp_df.shape[0] != 25:
            continue
        temp_df = temp_df.iloc[1:25]
        #UPDATING MODELS FOR EVERY COLUMN
        data1 = list(temp_df.iloc[:,3].values)
        LSTM_model_update(data1,key,'field_volume')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()

        data2 = list(temp_df.iloc[:,5].values)
        LSTM_model_update(data2,key,'hcv')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()
        
        data3 = list(temp_df.iloc[:,7].values)
        LSTM_model_update(data3,key,'pressure')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()
        
        data4 = list(temp_df.iloc[:,8].values)
        LSTM_model_update(data4,key,'temperature')
        progress_bars[counter]['value'] += 25
        percentage_labels[counter].config(text='{perc}%'.format(perc = progress_bars[counter]['value']))
        progress_window.update()

    progress_window.destroy()
    for file in glob.glob("*.csv"):
        remove(file)

def command3(gas_station):
    main_window.grid_columnconfigure((0,1), weight=1, uniform="column")
    main_window.grid_rowconfigure((0,1), weight=1, uniform="row")

    preds1,targets1 = pickle.load(open('models\{gas_station_name}_field_volume_preds.sav'.format(gas_station_name = gas_station),"rb"))
    max_err1 = max_ape(preds1,targets1)[0]
    mean_err1 = mean_ape(preds1,targets1)[0]
    if max_err1 == 0xffffffff or mean_err1 == 0xffffffff:
        text1 = gas_station.upper() + "\nFIELD VOLUME\nMax Error: infinity\nMean Error: infinity"
    else:
        text1 = gas_station.upper() + "\nFIELD VOLUME\nMax Error: {:.2f}%\nMean Error: {:.2f}%".format(max_err1,mean_err1)
    Button(main_window,text=text1,command=partial(command5,preds1,targets1,gas_station,'field volume'),background="black",font="Cambria 16 italic bold",foreground="white").grid(row=0,column=0,sticky=E+W+N+S)

    preds2,targets2 = pickle.load(open('models\{gas_station_name}_hcv_preds.sav'.format(gas_station_name = gas_station),"rb"))
    max_err2 = max_ape(preds2,targets2)[0]
    mean_err2 = mean_ape(preds2,targets2)[0]
    if max_err2 == 0xffffffff or mean_err2 == 0xffffffff:
        text2 =  gas_station.upper() + "\nHCV\nMax Error: infinity\nMean Error: infinity"
    else:
        text2 =  gas_station.upper() + "\nHCV\nMax Error: {:.2f}%\nMean Error: {:.2f}%".format(max_err2,mean_err2)
    Button(main_window,text=text2,command=partial(command5,preds2,targets2,gas_station,'hcv'),background="black",font="Cambria 16 italic bold",foreground="white").grid(row=0,column=1,sticky=E+W+N+S)
    
    preds3,targets3 = pickle.load(open('models\{gas_station_name}_pressure_preds.sav'.format(gas_station_name = gas_station),"rb"))
    max_err3 = max_ape(preds3,targets3)[0]
    mean_err3 = mean_ape(preds3,targets3)[0]
    if max_err3 == 0xffffffff or mean_err3 == 0xffffffff:
        text3 = gas_station.upper() + "\nPRESSURE\nMax Error: infinity\nMean Error: infinity"
    else:
        text3 = gas_station.upper() + "\nPRESSURE\nMax Error: {:.2f}%\nMean Error: {:.2f}%".format(max_err3,mean_err3)
    Button(main_window,text=text3,command=partial(command5,preds3,targets3,gas_station,'pressure'),background="black",font="Cambria 16 italic bold",foreground="white").grid(row=1,column=0,sticky=E+W+N+S)
    
    preds4,targets4 = pickle.load(open('models\{gas_station_name}_temperature_preds.sav'.format(gas_station_name = gas_station),"rb"))
    max_err4 = max_ape(preds4,targets4)[0]
    mean_err4 = mean_ape(preds4,targets4)[0]
    if max_err4 == 0xffffffff or mean_err4 == 0xffffffff:
        text4 = gas_station.upper() + "\nTEMPERATURE\nMax Error: infinity\nMean Error: infinity"
    else:
        text4 = gas_station.upper() + "\nTEMPERATURE\nMax Error: {:.2f}%\nMean Error: {:.2f}%".format(max_err4,mean_err4)
    Button(main_window,text=text4,command=partial(command5,preds4,targets4,gas_station,'temperature'),background="black",font="Cambria 16 italic bold",foreground="white").grid(row=1,column=1,sticky=E+W+N+S)
    

    
def command4(gas_station):
    for file in glob.glob("models/{gas_station}*".format(gas_station=gas_station)):
        if(isfile(file)):
            remove(file)
        elif(isdir(file)):
            shutil.rmtree(file)
    import_options_menu.delete(gas_station)
    delete_menu.delete(gas_station)
    gas_stations_list.remove(gas_station)

def command5(preds,targets,gas_station,column):
    
    fig = plt.Figure(figsize = (5,5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(preds,label="predicted values")
    plot1.plot(targets,label="actual data")
    plot1.set_title(gas_station+' - '+column)
    fig.legend(loc ="lower left")
    pop_up_window = Toplevel()
    canvas = FigureCanvasTkAgg(fig,master=pop_up_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 0,column = 0,sticky=E+W+N+S)
    

def max_ape(preds,targets):
  if len(preds) != len(targets):
    print("MAX_APE WARNING: Lists of incompatible size.")
    return
  max_error = 0
  for pred, target in zip(preds, targets):
    if target == 0:
      return [0xffffffff]
    err = abs(pred - target)/target
    if err > max_error:
      max_error = err
  return max_error*100

def mean_ape(preds,targets):
    if len(preds) != len(targets):
        print("MAX_APE WARNING: Lists of incompatible size.")
    if 0 in targets:
        return [0xffffffff]
    sum = 0
    for i in range(len(preds)):
        sum += abs(preds[i]-targets[i])/targets[i]
    
    return sum/len(preds)*100

def reset_all():
    for widgets in main_window.winfo_children():
        if type(widgets) == Button:              
            widgets.destroy()
    
gas_stations_list = []
for file in glob.glob("models/*_temperature.sav"):
    gas_stations_list.append(file[file.find('\\')+1:file.find('_')])
main_window = Tk()
main_window.title("Gas station data processing")
main_window.configure(background="black")

photo = PhotoImage(file='gas.PNG')
main_window.geometry("{width}x{width}".format(height = photo.height(), width = photo.width()))
main_window.resizable(False,False)

my_menu = Menu(main_window)
main_window.config(menu=my_menu)

file_menu = Menu(my_menu)
my_menu.add_cascade(label='File',menu = file_menu)

file_menu.add_command(label='New gas stations',command= command1)
file_menu.add_command(label='Existing gas stations update',command= command2)
file_menu.add_separator()
file_menu.add_command(label='Exit',command= close_window)

import_menu = Menu(my_menu)
my_menu.add_cascade(label='Import',menu = import_menu)

import_options_menu = Menu(import_menu)
import_menu.add_cascade(label='Import gas station',menu=import_options_menu)

edit_menu = Menu(my_menu)
my_menu.add_cascade(label='Edit',menu = edit_menu)
delete_menu = Menu(edit_menu)
edit_menu.add_cascade(label='Delete gas station',menu = delete_menu)

reset_menu = Menu(my_menu)
my_menu.add_cascade(label='Reset',menu = reset_menu)
reset_menu.add_command(label='Reset window',command=reset_all)

Label(main_window,image=photo,bg="black").place(x=0,y=0,relheight=1,relwidth=1)

for gas_station in gas_stations_list:
    import_options_menu.add_command(label=gas_station,command=partial(command3,gas_station))
    if not exists('models\{gas_station}_hcv_preds.sav'.format(gas_station = gas_station)):
        import_options_menu.entryconfig(gas_station, state="disabled")
    delete_menu.add_command(label=gas_station,command=partial(command4,gas_station))

main_window.mainloop()