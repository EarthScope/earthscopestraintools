import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

#from earthscopestraintools.gtsm_metadata import GtsmMetadata
import logging

logger = logging.getLogger(__name__)

def map_strain(areal:np.array,
               differential:np.array,
               shear:np.array,
               lats:np.array=None,
               lons:np.array=None,
               station_list:list=None,
               colors:list=None,
               units:str=None,
               scale_length:float=None,
               title:str=None,
               savefig:str=None):
    """Creates a map of the principal strain axes (x-axis = East, y-axis = North). Supply area, differential, and shear strains with optional location information and plot parameters.
       If a plot with strain units on the x- and y- axes is desired, omit station location information and the strain axes will be centered on zero. 
    
    :param areal: Array of areal (Eee+Enn) strains
    :type areal: np.array
    :param differential: Array of differential (Eee-Enn) strains
    :type differential: np.array
    :param shear: Array of engineering shear (2Een) strains
    :type shear: np.array
    :param lats: (Optional) Array of latitudes or y-axis locations
    :type lats: np.array
    :param lons: (Optional) Array of longitudes or y-axis locations
    :type lons: np.array
    :param station_list: (Optional) List of station names to label.
    :type station_list: list
    :param colors: (Optional) List of colors for the strain axes
    :type colors: list
    :param units: (Optional) Units to label strain
    :type units: str 
    :param scale_length: (Optional) Length to make the scale bar. Will plot a default scale based on magnitude of the data none is provided.
    :type scale_length: float
    :param title: Plot title
    :type title: str
    :param savefig: Save figure with the input string as the filename. Extension will determine format (png, svg, pdf, eps, ps).
    :type savefig: str
    :return: Map/Figure of strain axes
    :rtype: matplotlib.pyplot.figure

    Example
    --------
    >>> from earthscopestraintools.strain_visualization import map_strain
    >>> import numpy as np
    >>> # Array of offsets to plot in areal, differential, and shear strains
    >>> ea = np.array([1,0.5,0.25])*10
    >>> ed = np.array([1,0.5,0])*10
    >>> es = np.array([0.5,-0.5,0])*10
    >>> # Colors for each set of axes
    >>> colors = ['lightblue','orange','maroon']
    >>> # labeels for each set of arrows
    >>> station_list = ['lightblue','orange','maroon']
    >>> # location informtion
    >>> lats, lons = np.array([13,12.5,12.0]), np.array([40,40.5,41])
    >>> # call the function
    >>> map_strain(areal=ea,differential=ed,shear=es,
    ...            lats=lats, lons=lons,station_list=station_list,
    ...            colors=colors,units='ms',title='Principal Strain Axes')
    >>>
    """
    if lats is None:
        lats = np.zeros(len(areal))
    if lons is None:
        lons = np.zeros(len(areal))
        scale = 1
    else:
        scale = (max(lons)-min(lons))*0.05

    eEE = 1/2*(areal+differential) # 1/2 [(eEE+eNN)+(eEE-eNN)]
    eNN = 1/2*(areal-differential) # 1/2 [(eEE+eNN)-(eEE-eNN)]
    eEN = 1/2*shear # 1/2 [2EN]
    
    avg_length = np.average(np.abs([eEE,eNN,eEN]))
    hw = 0.2*avg_length * scale
    lw = 0.05*avg_length * scale
    
    fig, ax = plt.subplots(figsize=(5,5))

    if isinstance(title,str):
        ax.set_title(title)

    for i in range(0,len(lats)):
        if colors is None:
            c = 'black'
        else:
            c = colors[i]
            
        eig1 = np.linalg.eig(np.array([[eEE[i],eEN[i]],
                                      [eEN[i],eNN[i]]]))
        for j in range(0,2):
            e1 = eig1[0][j]*eig1[1][:,j]*scale      
            if eig1[0][j] < 0:
                ax.arrow(lons[i]+e1[0],lats[i]+e1[1],-e1[0],-e1[1],head_starts_at_zero=False,zorder=3,color=c,length_includes_head=True,head_width=hw,width=lw)
                ax.arrow(lons[i]-e1[0],lats[i]-e1[1],e1[0],e1[1],head_starts_at_zero=False,zorder=3,color=c,length_includes_head=True,head_width=hw,width=lw)
            else:
                ax.arrow(lons[i],lats[i],-e1[0],-e1[1],zorder=3,color=c,length_includes_head=True,head_width=hw,width=lw)
                ax.arrow(lons[i],lats[i],e1[0],e1[1],zorder=3,color=c,length_includes_head=True,head_width=hw,width=lw)

        # Station text
        if isinstance(station_list,list):
            t = ax.text(lons[i],lats[i],station_list[i],color='darkblue')
            t.set_bbox(dict(facecolor='lightblue', alpha=0.8, edgecolor='grey'))
        
    # Aspect for map plotting at specified latitude           
    f = 1.0/np.cos(np.average(lats)*np.pi/180)
    ax.set_aspect(f)
    ax.set_adjustable('datalim')

    # Arrow scale
    xax_length = ax.get_xlim()[1] - ax.get_xlim()[0]
    yax_length = ax.get_ylim()[1] - ax.get_ylim()[0]
    scale_origin = [ax.get_xlim()[0]+xax_length*0.15,
                    ax.get_ylim()[0]]
    if scale_length is None:
        scale_length = np.round(avg_length*2/3*scale,3)
    if units is None:
        units = ''
    else:
        units = f'[{units}]'
    ax.arrow(scale_origin[0],scale_origin[1],scale_length,0,head_starts_at_zero=False,zorder=3,color='red',length_includes_head=True,head_width=hw*0.6,width=lw*0.6)
    ax.arrow(scale_origin[0],scale_origin[1],-scale_length,0,head_starts_at_zero=False,zorder=3,color='red',length_includes_head=True,head_width=hw*0.6,width=lw*0.6)
    t = ax.text(scale_origin[0],scale_origin[1]+0.07*yax_length,s=str(scale_length)+' '+units,horizontalalignment='center',verticalalignment='center',color='indigo')
    t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='pink'))
    ax.set_xlabel(f'East {units}')
    ax.set_ylabel(f'North {units}')
    
    if isinstance(savefig,str):
        print('Saving Figure:',savefig)
        fig.savefig(savefig,dpi=300)

    return

def strain_video(df,
                 start:str=None,
                 end:str=None,
                 skip:int=1,
                 interval:float=None,
                 title:str=None,
                 units:str=None,
                 repeat:bool=False,
                 savegif:str=None):
    """Displays a gif of the strain time series provided, with time series and strain axes displayed. Strain is shown relative to the first data point. 

    :param df: Dataframe with datetime (in seconds) index and regional strain columns ('Eee+Enn', 'Eee-Enn', '2Ene')
    :type df: pd.DataFrame
    :param start: (Optional) Start of the video as a datetime string.
    :type start: str
    :param end: (Optional) End of the video as a datetime string.
    :type end: str
    :param skip: (optional) number of data points to skip per frame (eg. if using 5 minute Timeseries, skip=2 will decimate the dataset to a 10 minute period)
    :type skip: int
    :param interval: (Optional) Time between frames (in microseconds). 
    :type interval:
    :param title: (Optional) Plot title
    :type title: str
    :param repeat: (Optional) Choose if the animation repeats. Defaults to false.
    :type repeat: bool
    :param units: (Optional) Units to label strain
    :type units: str 
    :return: Gif of the strain time series
    :rtype: matplotlib.animation
    """
    if start == None:
        start = df.index[0]
    if end == None:
        end = df.index[-1]

    areal, differential, shear = df[start:end]['Eee+Enn'].values[::skip], df[start:end]['Eee-Enn'].values[::skip], df[start:end]['2Ene'].values[::skip]
    time = df[start:end].index[::skip]
    e = 1/2*(areal+differential) # 1/2 [(eEE+eNN)+(eEE-eNN)]
    eEE = e - e[0]
    n = 1/2*(areal-differential) # 1/2 [(eEE+eNN)-(eEE-eNN)]
    eNN = n - n[0]
    en = 1/2*shear # 1/2 [2EN]
    eEN = en - en[0]
    
    avg_length = np.average(np.abs([eEE,eNN,eEN]))
    hw = 0.2*avg_length
    lw = 0.05*avg_length

    plt.rcParams["animation.html"] = "jshtml"
    plt.close('all')

    # Start Figure
    fig = plt.figure()
    fig.suptitle(title)
    # First axis for principal strains
    ax = plt.subplot2grid((4,2),(0,0),rowspan=3,colspan=2)
    # Second axis for time series
    ax2 = plt.subplot2grid((4,2),(3,0),rowspan=1,colspan=2)
    # Aspect equal  
    ax.set_aspect('equal')
    ax.set_xlim(-max(np.abs(eEE)+np.abs(eNN)),max(np.abs(eEE)+np.abs(eNN)))
    ax.set_ylim(-max(np.abs(eEE)+np.abs(eNN)),max(np.abs(eEE)+np.abs(eNN)))

    if units is None:
        units = ''
    else:
        units = f'[{units}]'
    ax.set_xlabel('East '+units)
    ax.set_ylabel('North '+units)

    ax2.plot(time,areal-areal[0],c='blue',label='Eee+Enn')
    ax2.plot(time,differential-differential[0],c='orange',label='Eee-Enn')
    ax2.plot(time,shear-shear[0],c='red',label='2Een')
    ax2.legend(loc='lower right')
    ax2.set_ylabel('Strain '+units)
    ax2.set_xlabel('Time')
    
    fig.tight_layout(h_pad=0.1)
    
    # Use init_function for artists that need to redraw on each frame
    line = ax2.axvline(x=time[0],alpha=0.5)
    a1 = ax.arrow(0,0,-0,-0,zorder=3,width=lw,color='black',head_width=hw,length_includes_head=True,)
    a2 = ax.arrow(-0,-0,0,0,zorder=3,width=lw,color='black',head_width=hw,length_includes_head=True)
    a3 = ax.arrow(0,0,-0,-0,zorder=3,width=lw,color='black',head_width=hw,length_includes_head=True)
    a4 = ax.arrow(0,0,0,0,zorder=3,width=lw,color='black',head_width=hw,length_includes_head=True)
    
    def animate(j):
        # Moving line with time
        line.set_data([time[j],time[j]], [0,1])
        eig1 = np.linalg.eig(np.array([[eEE[j],eEN[j]],
                                      [eEN[j],eNN[j]]]))
        # Arrows for each eigenvalue
        i = 0
        e1 = eig1[0][i]*eig1[1][:,i]
        if eig1[0][i] < 0:
            a1.set_data(x=e1[0],y=e1[1],dx=-e1[0],dy=-e1[1])
            a2.set_data(x=-e1[0],y=-e1[1],dx=e1[0],dy=e1[1])
        else:
            a1.set_data(x=0,y=0,dx=-e1[0],dy=-e1[1])
            a2.set_data(x=0,y=0,dx=e1[0],dy=e1[1])
        i = 1
        e1 = eig1[0][i]*eig1[1][:,i]
        if eig1[0][i] < 0:
            a3.set_data(x=e1[0],y=e1[1],dx=-e1[0],dy=-e1[1])
            a4.set_data(x=-e1[0],y=-e1[1],dx=e1[0],dy=e1[1])
        else:
            a3.set_data(x=0,y=0,dx=-e1[0],dy=-e1[1])
            a4.set_data(x=0,y=0,dx=e1[0],dy=e1[1])
        return 
        
    if interval == None:
        interval = 200
    anim = matplotlib.animation.FuncAnimation(fig=fig, func=animate,frames=len(time),interval=interval,repeat=repeat)
    if isinstance(savegif,str):
        writergif = matplotlib.animation.PillowWriter() 
        print('Saving Figure:',savegif)
        anim.save(savegif,writer=writergif,dpi=300)
    plt.show()
    return anim


