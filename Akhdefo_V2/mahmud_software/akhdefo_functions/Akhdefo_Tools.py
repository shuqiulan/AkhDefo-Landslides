def utm_to_latlon(easting, northing, zone_number, zone_letter):
    '''
    This program converts geographic projection of shapefiles from UTM to LATLONG
    
    Parameters
    ----------
    easting: Geopandas column with Easting 
    
    northing: Geopandas column with Northing
    
    zone_number: int
    
    zone_letter: "N" or "S"
    
    Returns
    -------
    [lon , lat ]: List

    '''
    import geopandas as gpd
    import utm
    easting = easting
    northing = northing
    lon, lat=utm.to_latlon(easting, northing, zone_number, zone_letter)
    
    return [lon, lat]

# def ts_plot(df, plot_number, save_plot=False , output_dir="", plot_filename="" , VEL_Scale='year'):


#     import plotly.graph_objects as go
#     import plotly.express as px
#     import plotly.express as px_temp
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import geopandas as gpd 
#     import pandas as pd  
#     import seaborn as sns  
#     import plotly.offline as py_offline
#     import os   
#     import statsmodels.api as sm
#     from sklearn.metrics import mean_squared_error, r2_score
#     import numpy as np
#     from sklearn.linear_model import LinearRegression
#     from datetime import datetime
#     import math
    
#     py_offline.init_notebook_mode()
#     #%matplotlib widget
#     #df=pd.read_csv("temp.csv")
#     df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
#     df['dd_str']=df['dd'].astype(str)
#     df['dd_str'] = df['dd_str'].astype(str)
#     df.rename(columns={ df.columns[1]: "val" }, inplace = True)
#     df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
    
#     df=df.set_index('dd')
    
#     ########################
#     df=df.dropna()
#     # Make index pd.DatetimeIndex
#     df.index = pd.DatetimeIndex(df.index)
#     # Make new index
#     idx = pd.date_range(df.index.min(), df.index.max())
#     # Replace original index with idx
#     df = df.reindex(index = idx)
#     # Insert row count
#     df.insert(df.shape[1],
#             'row_count',
#             df.index.value_counts().sort_index().cumsum())

#     df=df.dropna()
    
#     #df=df.set_index(df['row_count'], inplace=True)

#     df.sort_index(ascending=True, inplace=True)
    

#     def best_fit_slope_and_intercept(xs,ys):
#         from statistics import mean
#         xs = np.array(xs, dtype=np.float64)
#         ys = np.array(ys, dtype=np.float64)
#         m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
#             ((mean(xs)*mean(xs)) - mean(xs*xs)))
        
#         b = mean(ys) - m*mean(xs)
        
#         return m, b

    

#     #convert dattime to number of days per year
    
    
    

#     dates_list=([datetime.strptime(x, '%Y%m%d') for x in df.dd_str])
#     days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
#     time2=days_num[len(days_num)-1]
#     time1=days_num[0]
#     delta=time2-time1
#     delta=float(delta)
#     print(days_num, delta)
    
#     m, b = best_fit_slope_and_intercept(df.row_count, df.val)
#     print("m:", math.ceil(m*100)/100, "b:",math.ceil(b*100)/100)
#     regression_model = LinearRegression()
#     val_dates_res = regression_model.fit(np.array(days_num).reshape(-1,1), np.array(df.val))
#     y_predicted = regression_model.predict(np.array(days_num).reshape(-1,1))
    
#     if VEL_Scale=='year':
#         rate_change=regression_model.coef_[0]/delta * 365.0
#     elif VEL_Scale=='month':
#         rate_change=regression_model.coef_[0]/delta * 30
        
#     # model evaluation
#     mse=mean_squared_error(np.array(df.val),y_predicted)
#     rmse = np.sqrt(mean_squared_error(np.array(df.val), y_predicted))
#     r2 = r2_score(np.array(df.val), y_predicted)
    
#     # printing values
#     print('Slope(linear deformation rate):' + str(math.ceil(regression_model.coef_[0]*100)/100/delta) + " mm/day")
#     print('Intercept:', math.ceil(b*100)/100)
#     #print('MSE:',mse)
#     print('Root mean squared error: ', math.ceil(rmse*100)/100)
#     print('R2 score: ', r2)
#     print("STD: ",math.ceil(np.std(y_predicted)*100)/100) 
#     # Create figure
#     #fig = go.Figure()
    
#     fig = go.FigureWidget()
    
#     plot_number="Plot Number:"+str(plot_number)

#     fig.add_trace(go.Scatter(x=list(df.index), y=list(df.val)))
#     fig = px.scatter(df, x=list(df.index), y=list(df.val),
#                 color="val", hover_name="val"
#                     , labels=dict(x="Dates", y="mm/"+VEL_Scale , color="mm/"+VEL_Scale))
    
#     # fig.add_trace(
#     # go.Scatter(x=list(df.index), y=list(val_fit), mode = "lines",name="trendline", marker_color = "red"))
    
    
    
#     fig.add_trace(go.Scatter(x=list(df.index), y=list(df.val),mode = 'lines',
#                             name = 'draw lines', line = dict(shape = 'linear', color = 'rgb(0, 0, 0)', dash = 'dash'), connectgaps = True))
    
#     fig.add_trace(
#         go.Scatter(x=list(df.index), y=list(y_predicted), mode = "lines",name="trendline", marker_color = "black", line_color='red'))
    
    

#     # Add range slider
#     fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1,
#                         label="1m",
#                         step="month",
#                         stepmode="backward"),
#                     dict(count=6,
#                         label="6m",
#                         step="month",
#                         stepmode="backward"),
#                     dict(count=1,
#                         label="YTD",
#                         step="year",
#                         stepmode="todate"),
#                     dict(count=1,
#                         label="1y",
#                         step="year",
#                         stepmode="backward"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         ) 
#     )
#     fig.update_xaxes(rangeslider_thickness = 0.05)
#     #fig.update_layout(showlegend=True)

#     #fig.data[0].update(line_color='black')
#     tt= "Defo-Rate:"+str(round(rate_change,2))+":"+ "Defo-Rate-STD:"+str(round(np.std(y_predicted), 2))+ ":" +plot_number
    
#     # make space for explanation / annotation
#     fig.update_layout(margin=dict(l=20, r=20, t=20, b=60),paper_bgcolor="LightSteelBlue")

    
#     fig.update_layout(
        
#     title_text=tt, title_font_family="Sitka Small",
#     title_font_color="red", title_x=0.5 , legend_title="Legend",
#     font=dict(
#         family="Courier New, monospace",
#         size=15,
#         color="RebeccaPurple" ))
    
#     fig.update_layout(legend=dict(
#     yanchor="top",
#     y=-0,
#     xanchor="left",
#     x=1.01
# ))

#     # fig.update_layout(
#     # updatemenus=[
#     #     dict(
#     #         type="buttons",
#     #         direction="right",
#     #         active=0,
#     #         x=0.57,
#     #         y=1.2,
#     #         buttons=list([
#     #             dict(
#     #                 args=["colorscale", "Viridis"],
#     #                 label="Viridis",
#     #                 method="restyle"
#     #             ),
#     #             dict(
#     #                 args=["colorscale", "turbo"],
#     #                 label="turbo",
#     #                 method="restyle"
#     #             )
#     #         ]),
#     #     )
#     # ])

    
#     fig.update_xaxes(showspikes=True, spikemode='toaxis' , spikesnap='cursor', spikedash='dot', spikecolor='blue', scaleanchor='y', title_font_family="Arial", 
#                     title_font=dict(size=15))
#     fig.update_yaxes(showspikes=True, spikemode='toaxis' , spikesnap='cursor', spikedash='dot', spikecolor='blue', scaleanchor='x', title_font_family="Arial",
#                     title_font=dict(size=15))

    
    
#     if save_plot==True:
    
#         if not os.path.exists(output_dir):
#             os.mkdir(output_dir)

#         fig.write_html(output_dir + "/" + plot_filename + ".html" )
#         fig.write_image(output_dir + "/" + plot_filename + ".jpeg", scale=1, width=1080, height=300 )
        
    
#     def zoom(layout, xrange):
#         in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
#         fig.layout.yaxis.range = [in_view.High.min() - 10, in_view.High.max() + 10]

#     fig.layout.on_change(zoom, 'xaxis.range')
    
#     fig.show()
    
    




    
#     start=int(start.timestamp() * 1000)
#     end=int(end.timestamp() * 1000)

#     #df=pd.read_csv('temp2.csv')
    
#     df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
#     df['dd_str']=df['dd'].astype(str)
#     df['dd_str'] = df['dd_str'].astype(str)
#     df.rename(columns={ df.columns[1]: "val" }, inplace = True)
#     df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y-%m-%d')
#     df.insert(df.shape[1],
#             'row_count',
#             df.index.value_counts().sort_index().cumsum())
#     #df=df.set_index('dd')
#     #df.index = pd.DatetimeIndex(df.index)
#     df.dd_str = pd.DatetimeIndex(df.dd_str)
#     df['dd_int'] = [int(i.timestamp()*1000) for i in df.dd_str]
#     import numpy as np 
#     def find_nearest(array, value):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return array[idx]
#     s=find_nearest(np.array(df.dd_int), start)
#     e=find_nearest(np.array(df.dd_int), end)

#     s=(df[df['dd_int']==s].index)
#     e=(df[df['dd_int']==e].index)

#     df_filter=df[s[0]:e[0]]
#     print(df_filter)

#     df=df_filter  
    
# import pandas as pd
# import ipywidgets as widgets
# from IPython.display import display

# class DateRangePicker(object):
#     def __init__(self,start,end,freq='D',fmt='%Y-%m-%d'):
#         """
#         Parameters
#         ----------
#         start : string or datetime-like
#             Left bound of the period
#         end : string or datetime-like
#             Left bound of the period
#         freq : string or pandas.DateOffset, default='D'
#             Frequency strings can have multiples, e.g. '5H' 
#         fmt : string, defauly = '%Y-%m-%d'
#             Format to use to display the selected period

#         """
#         self.date_range=pd.date_range(start=start,end=end,freq=freq)
#         options = [(item.strftime(fmt),item) for item in self.date_range]
#         self.slider_start = widgets.SelectionSlider(
#             description='start',
#             options=options,
#             continuous_update=False
#         )
#         self.slider_end = widgets.SelectionSlider(
#             description='end',
#             options=options,
#             continuous_update=False,
#             value=options[-1][1]
#         )

#         self.slider_start.on_trait_change(self.slider_start_changed, 'value')
#         self.slider_end.on_trait_change(self.slider_end_changed, 'value')

#         self.widget = widgets.Box(children=[self.slider_start,self.slider_end])

#     def slider_start_changed(self,key,value):
#         self.slider_end.value=max(self.slider_start.value,self.slider_end.value)
#         self._observe(start=self.slider_start.value,end=self.slider_end.value)

#     def slider_end_changed(self,key,value):
#         self.slider_start.value=min(self.slider_start.value,self.slider_end.value)
#         self._observe(start=self.slider_start.value,end=self.slider_end.value)

#     def display(self):
#         display(self.slider_start,self.slider_end)

#     def _observe(self,**kwargs):
#         if hasattr(self,'observe'):
#             self.observe(**kwargs)

# def fct(start,end):
#     print (start,end)
    
#     start=int(start.timestamp() * 1000)
#     end=int(end.timestamp() * 1000)

#     df=pd.read_csv('temp2.csv')

#     df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
#     df['dd_str']=df['dd'].astype(str)
#     df['dd_str'] = df['dd_str'].astype(str)
#     df.rename(columns={ df.columns[1]: "val" }, inplace = True)
#     df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y-%m-%d')
#     df.insert(df.shape[1],
#             'row_count',
#             df.index.value_counts().sort_index().cumsum())
#     #df=df.set_index('dd')
#     #df.index = pd.DatetimeIndex(df.index)
#     df.dd_str = pd.DatetimeIndex(df.dd_str)
#     df['dd_int'] = [int(i.timestamp()*1000) for i in df.dd_str]
#     import numpy as np 
#     def find_nearest(array, value):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return array[idx]
#     s=find_nearest(np.array(df.dd_int), start)
#     e=find_nearest(np.array(df.dd_int), end)

#     s=(df[df['dd_int']==s].index)
#     e=(df[df['dd_int']==e].index)

#     df_filter=df[s[0]:e[0]]
#     print(df_filter)
#     return (start, end)
    
# w=DateRangePicker(start='2022-08-02',end="2022-09-02",freq='D',fmt='%Y-%m-%d')
# w.observe=fct
# w.display()

# #a=fct[0]
# print(w.observe[0])