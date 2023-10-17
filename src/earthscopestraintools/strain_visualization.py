import numpy as np
import matplotlib.pyplot as plt

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
    '''Creates a map of the principal strain axes. Supply area, differential, and shear strains with optional location information.
    :param areal: Array of areal (Eee+Enn) strains
    :type areal: np.array
    :param differential: Array of differential (Eee-Enn) strains
    :type differential: np.array
    :param shear: Array of engineering shear (2Een) strains
    :type shear: np.array
    :param lats: (Optional) Array of latitudes or y-axis locations
    :type lats:np.array
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
    '''
    
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
            t = ax.text(lons[i],lats[i],station_list[i])
            t.set_bbox(dict(facecolor='grey', alpha=0.3, edgecolor='grey'))
        
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
    ax.arrow(scale_origin[0],scale_origin[1],scale_length,0,head_starts_at_zero=False,zorder=3,color='orange',length_includes_head=True,head_width=hw*0.6,width=lw*0.6)
    ax.arrow(scale_origin[0],scale_origin[1],-scale_length,0,head_starts_at_zero=False,zorder=3,color='orange',length_includes_head=True,head_width=hw*0.6,width=lw*0.6)
    t = ax.text(scale_origin[0],scale_origin[1]+0.1*yax_length,s=str(scale_length)+' '+units,horizontalalignment='center',verticalalignment='center')
    t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    
    if isinstance(savefig,str):
        print('Saving Figure:',savefig)
        fig.savefig(savefig,dpi=300)


