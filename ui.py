# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:45:12 2022

@author: Hugo Xue
@email: hugo@wustl.edu; 892849924@qq.com

"""

import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg
from tkinter.ttk import Progressbar
from pandastable import Table
from tkintertable import TableCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from Forecast import RoyalCanin
import time
import threading


class exeFC:
   
    def __init__(self, root):
   
        self.root = root
        self.file_name = ''
        self.f = Frame(self.root,
                       height = 200,
                       width = 500)
          
        # Place the frame on root window
        self.f.pack()
        
        self.selectedDC = StringVar()
        self.selectedSKU = StringVar()
        self.selectedModel = StringVar()
        
        # Creating label widgets
        self.message_label = Label(self.f,
                                   text = 'Royal Canin Forecasting Model',
                                   font = ('Arial', 19,'underline'),
                                   fg = 'Red')
        # input parameters
        self.seleDC_label = Label(self.f,
            text = 'Selected A DC you want to forecast (ex. ASC- DENVER)',
            font = ('Arial', 12),
            fg = 'Black')
        self.seleDC_entry = Entry(self.f, textvariable = self.selectedDC)
        
        self.seleSKU_label = Label(self.f,
            text = 'Selected A SKU you want to forecast (ex. 41006)',
            font = ('Arial', 12),
            fg = 'Black')
        self.seleSKU_entry = Entry(self.f, textvariable = self.selectedSKU)
        
        self.selectedModel_label = Label(self.f,
            text = 'Selected a model (ARIMA/LSTM/CNN)',
            font = ('Arial', 12),
            fg = 'Black')
        
        self.radio_button1 = Radiobutton(self.f, text='ARIMA (one-step)', 
                                       variable=self.selectedModel, value="arima")
        self.radio_button2 = Radiobutton(self.f, text='CNN (multi-step)', 
                                       variable=self.selectedModel, value="cnn")
        self.radio_button3 = Radiobutton(self.f, text='LSTM (multi-step)', 
                                       variable=self.selectedModel, value="lstm")
        
        # Buttons                
        self.forecast_button = Button(self.f,
                                     text = 'Forecast',
                                     font = ('Arial', 14),
                                     bg = 'Orange',
                                     fg = 'Black',
                                     command = self.run)

        self.exit_button = Button(self.f,
                                  text = 'Exit',
                                  font = ('Arial', 14),
                                  bg = 'Red',
                                  fg = 'Black', 
                                  command = root.destroy)
        
        self.read_file_button = Button(self.f, text='Import csv File',
                                       font = ('Arial', 14),
                                       bg = 'Blue',
                                       fg = 'Black', 
                                       command = self.read_file)
        
        self.display_button = Button(self.f,
                                     text = 'Display Charts',
                                     font = ('Arial', 14),
                                     bg = 'Green',
                                     fg = 'Black',
                                     command = self.display_file)
        
        self.clear_frame_button = Button(self.f,
                                     text = 'Clear Table',
                                     font = ('Arial', 14),
                                     fg = 'Black',
                                     command = self.clear_frame)
        self.clear_canvas_button = Button(self.f,
                                     text = 'Clear Figures',
                                     font = ('Arial', 14),
                                     fg = 'Black',
                                     command = self.clear_canvas)   
        
        # Placing the widgets using grid manager
        self.message_label.grid(row = 1, column = 0)

        self.seleDC_label.grid(row = 2, column = 0)
        self.seleDC_entry.grid(row = 2, column = 1)
        
        self.seleSKU_label.grid(row = 3, column = 0)
        self.seleSKU_entry.grid(row = 3, column = 1)
        
        self.radio_button1.grid(row = 4, column = 0)
        self.radio_button2.grid(row = 4, column = 1)
        self.radio_button3.grid(row = 4, column = 2)
        
        self.read_file_button.grid(row = 5, column = 0, sticky = E)
        self.forecast_button.grid(row = 6, column = 0, sticky = E)
        self.display_button.grid(row = 7, column = 0, sticky = E)
        self.exit_button.grid(row = 8, column = 0, sticky = E)
        self.clear_frame_button.grid(row = 8, column = 1, sticky = E)
        self.clear_canvas_button.grid(row = 8, column = 2, sticky = E)
        
        self.data = None 
        
        # the order transaction date: the label of date
        self.orderDate = 'gii__OrderDate__c'
        
        # the sku: the label of sku
        self.orderSKU = 'giic_Product_SKU__c'
        
        # the dc description: the label of dc
        self.orderDC = 'gii__Description__c'
        
        # the order quantity: the label we want to forecast
        self.orderQuantity = 'sum(gii__OrderQuantity__c)'
        
        self.df_result = None
        
        self.progress = Progressbar(self.f, orient=HORIZONTAL,
                                    length=100, mode="indeterminate")
        
        
    def read_file(self):
        """
        read csv file

        Returns
        -------
        None.

        """
        def readFile():
            self.progress.grid(row=0, column=0)
            self.progress.start()
            time.sleep(20)
            self.progress.stop()
            self.progress.grid_forget()
            
        try:
            self.file_name = filedialog.askopenfilename(initialdir = '/Desktop',
                                                        title = 'Select a CSV file',
                                                        filetypes = (('csv file','*.csv'),
                                                                     ('csv file','*.csv')))
            threading.Thread(target=readFile).start()
            
            self.data = pd.read_csv(self.file_name, index_col=0,
                       parse_dates=[self.orderDate])
            
        
            self.f3 = Frame(self.root, height=5, width=300) 
            self.f3.pack(fill=BOTH,expand=1)
            self.table = Table(self.f3, dataframe=self.data , read_only=True)
            self.table.show()
                   
        except FileNotFoundError as e:
                msg.showerror('Error in opening file', e)
        
    def run(self):
        """
        run forecasting model

        Returns
        -------
        None.

        """
        try:              
            # Next - Pandas DF to Excel file on disk
            if(len(self.data) == 0):      
                msg.showinfo('No Rows Selected', 'CSV has no rows')
            else:
                selected_DC_Data = self.data[self.data[self.orderDC] == self.selectedDC.get()]
                selected_SKU_Data = selected_DC_Data[selected_DC_Data[self.orderSKU] == self.selectedSKU.get()]
                
                # converted into weekly data
                wkly_data = selected_SKU_Data.resample('W-Mon', label='right', on=self.orderDate)\
                .sum().reset_index().sort_values(by=self.orderDate)
                wkly_data.set_index(self.orderDate, inplace=True)
                
                model = RoyalCanin(wkly_data, labels=[self.orderQuantity], freq=7, 
                                   dc=None, sku=None)
                
                model_mode = self.selectedModel.get()
                print(model_mode)
                
                model.fit(mode=model_mode, split_ratio=0.7, input_steps=12, output_steps=12, 
                          conv_width=3, max_epochs=20, patience=3, window=90)
    
                self.df_result = model.get_result(option = "dataframe")
                
                # # saves in the current directory
                self.df_result.to_csv('./result.csv')
                msg.showinfo('result.csv file created', 'result.csv File created')
                        
        except:
                msg.showinfo("Is everything all set?", "Is everything all set?")
    
    def clear_frame(self):
        """
        clear table function

        Returns
        -------
        None.

        """
        if self.f3:
            for widget in self.f3.winfo_children():
                widget.destroy()
            
            self.f3.pack_forget()
        else:
            pass
    
    def clear_canvas(self):
        """
        clear canvas / figure function

        Returns
        -------
        None.

        """
        if self.canvas:
            for item in self.canvas.get_tk_widget().find_all():
               self.canvas.get_tk_widget().delete(item)
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()
        else:
            pass
        
    def display_file(self):
        """
        display / figure

        Returns
        -------
        None.

        """
        try:
            self.file_name = filedialog.askopenfilename(initialdir = '/Desktop',
                                                        title = 'Select a CSV file',
                                                        filetypes = (('csv file','*.csv'),
                                                                     ('csv file','*.csv')))
            if self.selectedModel.get() == 'arima':
                result = pd.read_csv(self.file_name, index_col=0,
                           parse_dates=["datetime", "fc_datetime"])
                history_datetime = result["datetime"].dropna()
                history_df = result['true'].dropna()
                
                predict_df = result['predict'].dropna()
                forecast_date = result['fc_datetime'].dropna()
                
                fig = Figure(figsize=(15, 8), dpi=100)
                ax = fig.add_subplot()
                
                line1, = ax.plot(history_datetime, history_df, color='grey', linestyle='-', label="history_true")
                line2, = ax.plot(forecast_date, predict_df, color='b', linestyle='-', label="history_predict")
                line3, = ax.plot(forecast_date.values[-1], predict_df.values[-1], color='red', marker='.', label="future_predict")
                
                ax.set_title("Order Quantity for the next week: {0} | r-Squared: {1} | rmse: {2}".format(
                    predict_df.values[-1], result['rSquared'][0], result['rmse'][0]))
                ax.legend(loc='upper left')
                ax.set_xlabel("date")
                ax.set_ylabel("order quantity")
                
                self.canvas = FigureCanvasTkAgg(fig, master = self.root)  
                
                self.canvas.draw()
                
                self.canvas.mpl_connect("key_press_event", lambda event: print(f"you pressed {event.key}"))
                self.canvas.mpl_connect("key_press_event", key_press_handler)
              
                # placing the canvas on the Tkinter window
                self.canvas.get_tk_widget().pack(fill=BOTH, expand=1)
              
                # creating the Matplotlib toolbar
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
                self.toolbar.update()
              
                # placing the toolbar on the Tkinter window
                self.toolbar.pack(side=BOTTOM, fill=X)
                self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            else:
                result = pd.read_csv(self.file_name, index_col=0,
                           parse_dates=["datetime", "forecast_date"])
                
                history_datetime = result["datetime"].dropna()
                history_df = result['true'].dropna()
                predict_df = result['predict'].dropna()
                forecast_date = result['forecast_date'].dropna()
                forecast = result['forecast'].dropna()
                
                fig = Figure(figsize=(15, 8), dpi=100)
                ax = fig.add_subplot()
                
                line1, = ax.plot(history_datetime, history_df, color='grey', linestyle='-', label="history_true")
                line2, = ax.plot(history_datetime, predict_df, color='b', linestyle='-', label="history_predict")
                line3, = ax.plot(forecast_date, forecast, color='red', marker='.', linestyle='dashed', label="future_predict")
                
                ax.set_title("Order Quantity for the next week: {0} | r-Squared: {1} | rmse: {2}".format(
                    forecast[0], result['rSquared'][0], result['rmse'][0]))
                ax.legend(loc='upper left')
                ax.set_xlabel("date")
                ax.set_ylabel("order quantity")
                
                self.canvas = FigureCanvasTkAgg(fig, master = self.root)  
                
                self.canvas.draw()
                
                self.canvas.mpl_connect("key_press_event", lambda event: print(f"you pressed {event.key}"))
                self.canvas.mpl_connect("key_press_event", key_press_handler)
              
                # placing the canvas on the Tkinter window
                self.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
              
                # creating the Matplotlib toolbar
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
                self.toolbar.update()
              
                # placing the toolbar on the Tkinter window
                self.toolbar.pack()
                # self.canvas.get_tk_widget().pack(side=BOTTM, fill=BOTH, expand=1)
        except:
            msg.showinfo('Error in opening file or No model option selectd', 
                         'Error in opening file or No model option selectd')
        
if __name__ == "__main__":
    root = Tk()
    root.title('Royal Canin Forecasting Program')
    
    obj = exeFC(root)
    root.geometry('800x600')
    root.mainloop()



# ASC- MONROE 41155
# ASC- FIFE 584003
# ASC - BARTLETT 800075
# ASC- ORRVILLE 715285


















