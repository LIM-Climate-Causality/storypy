"""
storypy.compute._regres
=======================

Low-level utilities for spatial multiple linear regression (MLR).

The :class:`spatial_MLR` class wraps a statsmodels OLS regression
fitted at each gridpoint of a target xarray object, using model-wise
driver indices as predictors. Additional helpers support detrending,
standardisation, and diagnostic plotting.

This module is typically not used directly by end users; instead they
call :func:`storypy.compute._mlr.run_regression`, which drives the
workflow and uses :class:`spatial_MLR` internally.
"""

import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib as mpl
import random

class spatial_MLR(object):
    """
    Spatial multiple linear regression over model ensembles.

    The class prepares regression matrices from a set of driver indices
    and a target DataArray with a ``model`` dimension, then performs an
    OLS fit at each gridpoint using :mod:`statsmodels`.

    Typical usage
    -------------
    >>> from storypy.compute._regres import spatial_MLR
    >>> MLR = spatial_MLR()
    >>> MLR.regression_data(target, regressors, regressor_names)
    >>> MLR.perform_regression("./out", "pr")  # doctest: +SKIP

    Attributes
    ----------
    target : xarray.DataArray
        Target field for the regression (with ``model`` dimension).
    regression_y : ndarray
        Design matrix used for OLS (constant + regressors).
    regressors : ndarray
        Raw regressor values.
    rd_num : int
        Number of regressors including the constant term.
    regressor_names : list-like
        Names of the constant and driver regressors.
    """
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,variable,regressors,regressor_names):
        """
        Configure target and regressor matrices for the regression.

        Parameters
        ----------
        variable : xarray.DataArray or None
            Target variable with a ``model`` dimension (e.g. model-wise
            normalized precipitation change). If ``None``, only the
            regressors are stored (useful for quick checks).
        regressors : pandas.DataFrame
            Model-wise driver indices. Each row corresponds to a model,
            columns to individual regressors.
        regressor_names : sequence of str
            Names of the regressors. The first entry is typically the
            regression intercept (e.g. ``"MEM"`` for multi-model mean).

        Notes
        -----
        The design matrix ``regression_y`` is created by calling
        :func:`statsmodels.api.add_constant` on the regressor values.
        """
        self.target = variable
        regressor_indices = regressors
        self.regression_y = sm.add_constant(regressors.values)
        self.regressors = regressors.values
        self.rd_num = len(regressor_names)
        self.regressor_names = regressor_names

    #Regresion lineal
    def linear_regression(self,x):
        """
        Fit an OLS regression for a single gridpoint.

        Parameters
        ----------
        x : array-like
            Target values over models for one gridpoint.

        Returns
        -------
        tuple
            Regression coefficients (one per regressor).
        """
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.params[i] for i in range(self.rd_num)]
        return tuple(returns)

    def linear_regression_pvalues(self,x):
        """
        Compute p-values for each regressor at a single gridpoint.

        Parameters
        ----------
        x : array-like
            Target values over models.

        Returns
        -------
        tuple
            p-values associated with each regression coefficient.
        """
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.pvalues[i] for i in range(self.rd_num)]
        return tuple(returns)
    
    def linear_regression_R2(self,x):
        """
        Compute the coefficient of determination (R²) for a gridpoint.

        Parameters
        ----------
        x : array-like
            Target values over models.

        Returns
        -------
        float
            R² for the OLS fit.
        """
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        return res.rsquared
    
    def linear_regression_relative_importance(self,x):
        """
        Compute relative importance of regressors (LMG-like metric).

        Notes
        -----
        This method assumes an R interface is available via
        ``robjects.globalenv['rel_importance']``. If that call fails,
        a vector of zeros is returned.

        Parameters
        ----------
        x : array-like
            Target values over models.

        Returns
        -------
        tuple
            Relative importance values for each regressor (excluding
            the constant term).
        """
        y = self.regressors
        try:
            res = robjects.globalenv['rel_importance'](x,y)
            returns = [res[i] for i in range((len(res)))]
            #print(res)
            correct = True
        except:
            returns2 = [np.array([0.0]) for i in range(self.rd_num-1)]
            correct = False
        finally:
            if correct:
                #print('I should return ',returns)
                return tuple(returns)
            else:
                #print('I am returning 0')
                return tuple(returns2)


    def perform_regression(self,path,var): 
        """
        Perform spatial regression over all gridpoints and save diagnostics.

        The regression is run independently at each gridpoint using
        :meth:`linear_regression`, :meth:`linear_regression_pvalues`,
        :meth:`linear_regression_R2`, and
        :meth:`linear_regression_relative_importance`. The resulting
        coefficients, p-values, relative importance and R² fields are
        written to NetCDF files under::

            <path>/<var>/

        Parameters
        ----------
        path : str
            Base directory for writing NetCDF outputs.
        var : str
            Name of the target variable (used in filenames and some
            variable renaming logic).

        Returns
        -------
        None
            Results are written to disk.
        """
        target_var = xr.apply_ufunc(replace_nans_with_zero, self.target)
        results = xr.apply_ufunc(self.linear_regression,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_pvalues = xr.apply_ufunc(self.linear_regression_pvalues,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_R2 = xr.apply_ufunc(self.linear_regression_R2,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[]],
                                 vectorize=True,
                                 dask="parallelized")
        
        relative_importance = xr.apply_ufunc(self.linear_regression_relative_importance,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[] for i in range(self.rd_num-1)],
                                 vectorize=True,
                                 dask="parallelized")
      
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs = results[0].to_dataset()
            else:
                regression_coefs[self.regressor_names[i]] = results[i]
                
        print('This is regressor_coefs:',regression_coefs)
        if var == 'ua':
            regression_coefs = regression_coefs.rename({'ua':self.regressor_names[0]})
        elif var == 'sst':
            regression_coefs = regression_coefs.rename({'tos':self.regressor_names[0]})
        elif var == 'tas':
            regression_coefs = regression_coefs.rename({'tas':self.regressor_names[0]})
        elif var == 'pr':
            regression_coefs = regression_coefs.rename({'pr':self.regressor_names[0]})
        else:
            regression_coefs = regression_coefs.rename({var:self.regressor_names[0]})
        regression_coefs.to_netcdf(path+'/'+var+'/regression_coefficients.nc')
        
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs_pvalues = results_pvalues[0].to_dataset()
            else:
                regression_coefs_pvalues[self.regressor_names[i]] = results_pvalues[i]        
        if var == 'ua':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'ua':self.regressor_names[0]})
        elif var == 'sst':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'tos':self.regressor_names[0]})
        elif var == 'tas':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'tas':self.regressor_names[0]})
        elif var == 'pr':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'pr':self.regressor_names[0]})
        else:
            regression_coefs_pvalues = regression_coefs_pvalues.rename({var:self.regressor_names[0]})
        regression_coefs_pvalues.to_netcdf(path+'/'+var+'/regression_coefficients_pvalues.nc')
        
        for i in range(len(relative_importance)):
            if i == 0:
                relative_importance_values = relative_importance[0].to_dataset()
            else:
                relative_importance_values[self.regressor_names[1:][i]] = relative_importance[i]
                
        if var == 'ua':
            relative_importance_values = relative_importance_values.rename({'ua':self.regressor_names[1]})
        elif var == 'sst':
            relative_importance_values = relative_importance_values.rename({'tos':self.regressor_names[1]})
        elif var == 'tas':
            relative_importance_values = relative_importance_values.rename({'tas':self.regressor_names[1]})
        elif var == 'pr':
            relative_importance_values = relative_importance_values.rename({'pr':self.regressor_names[1]})
        else:
            relative_importance_values = relative_importance_values.rename({var:self.regressor_names[1]})
    
        relative_importance_values.to_netcdf(path+'/'+var+'/regression_coefficients_relative_importance.nc')
        results_R2.to_netcdf(path+'/'+var+'/R2.nc')
                     
        
    def create_x(self,i,j,dato):
        """ For each gridpoint creates an array and standardizes it 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """    
        x = np.array([])
        for y in range(len(dato.time)):
            aux = dato.isel(time=y)
            x = np.append(x,aux[i-1,j-1].values)
        return stand(x)
     
    
    def open_regression_coef(self,path,var):
        """
        Open saved regression coefficient and p-value maps.

        Parameters
        ----------
        path : str
            Base output path passed to :meth:`perform_regression`.
        var : str
            Target variable name.

        Returns
        -------
        (list[xarray.DataArray], list[xarray.DataArray], xarray.Dataset)
            Tuple ``(maps, maps_pval, R2)`` with coefficient maps,
            p-value maps, and R² dataset.
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/'+var+'/regression_coefficients.nc')
        coef_pvalues = xr.open_dataset(path+'/'+var+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/'+var+'/R2.nc')
        return maps, maps_pval, R2    

    def open_lmg_coef(self,path,var):
        """
        Open relative-importance regression maps.

        Parameters
        ----------
        path : str
            Base output path passed to :meth:`perform_regression`.
        var : str
            Target variable name.

        Returns
        -------
        (list[xarray.DataArray], list[xarray.DataArray], xarray.Dataset)
            Tuple ``(maps, maps_pval, R2)`` for LMG-style importance and
            the associated R² dataset.
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/'+var+'/regression_coefficients_relative_importance.nc')
        coef_pvalues = xr.open_dataset(path+'/'+var+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names[1:]]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/'+var+'/R2.nc')
        return maps, maps_pval, R2    
    
    def plot_regression_lmg_map(self,path,var,output_path):
        """
        Plot relative-importance maps for each regressor.

        Parameters
        ----------
        path : str
            Base path used when saving regression outputs.
        var : str
            Target variable name.
        output_path : str
            Directory where the figure will be saved.

        Returns
        -------
        matplotlib.figure.Figure
            Generated figure instance.
        """
        maps, maps_pval, R2 = self.open_lmg_coef(path,var)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num-1):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            clevels = np.arange(0,40,2)
            im=ax.contourf(lon_c, lat, var_c*100,clevels,transform=data_crs,cmap='OrRd',extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k+1].min() < 0.05: 
                levels = [maps_pval[k+1].min(),0.05,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k+1].min() < 0.10:
                levels = [maps_pval[k+1].min(),0.10,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k+1]) 
            plt.title(self.regressor_names[k+1],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.5, bottom, 0.01, height*2])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*2])    
        else:
            colorbar_axes1 = fig_coef.add_axes([left+0.5, bottom, 0.01, height*2])
        cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
        cbar.set_label('relative importance',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/regression_coefficients_relative_importance_u850',bbox_inches='tight')
        elif var == 'sst':
            plt.savefig(output_path+'/regression_coefficients_relative_importance_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/regression_coefficients_relative_importance_XXX',bbox_inches='tight')   
        plt.clf

        return fig_coef


    def plot_regression_coef_map(self,path,var,output_path):
        """
        Plot regression coefficient maps with significance hatching.

        Parameters
        ----------
        path : str
            Base path used when saving regression outputs.
        var : str
            Target variable name.
        output_path : str
            Directory where the figure will be saved.

        Returns
        -------
        matplotlib.figure.Figure
            Generated figure instance.
        """
        maps, maps_pval, R2 = self.open_regression_coef(path,var)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        # u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        # u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        # u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic for winds
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            #Plate Carree map for SST
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            if k == 0:
                im0=ax.contourf(lon_c, lat, var_c,transform=data_crs,cmap='OrRd',extend='both')
            else:
                clevels = np.arange(-.6,.7,0.1)
            #     im=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap='RdBu_r',extend='both')
            # cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            # plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k].min() < 0.05: 
                levels = [maps_pval[k].min(),0.05,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k].min() < 0.10:
                levels = [maps_pval[k].min(),0.10,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k]) 
            plt.title(self.regressor_names[k],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            if var == 'ua':
                ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            elif var == 'sst':
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            else: 
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.28, bottom, 0.01, height*2])
            colorbar_axes2 = fig_coef.add_axes([left+0.36, bottom, 0.01, height*2])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*3])
            colorbar_axes2 = fig_coef.add_axes([left+0.38, bottom, 0.01, height*3])
        cbar = fig_coef.colorbar(im0, colorbar_axes1, orientation='vertical')
        cbar2 = fig_coef.colorbar(im, colorbar_axes2, orientation='vertical')
        if var == 'ua':
            cbar.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
        elif var == 'sst':
            cbar.set_label('K/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('K/std(rd)',fontsize=14) #rotation = radianes
        else:
            cbar.set_label('X/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('X/std(rd)',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
        cbar2.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/regression_coefficients_u850',bbox_inches='tight')
        elif  var == 'sst':
            plt.savefig(output_path+'/regression_coefficients_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/regression_coefficients_unknown_var',bbox_inches='tight')
        
        plt.clf

        return fig_coef
    
def stand_detr(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return signal.detrend(anom)

def filtro(dato):
    """Apply a rolling mean of 5 years and remov the NaNs resulting bigining and end"""
    signal = dato - dato.rolling(time=10, center=True).mean()
    signal_out = signal.dropna('time', how='all')
    return signal_out
                          
def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom

def replace_nans_with_zero(x):
    return np.where(np.isnan(x), random.random(), x)

def figure(target,predictors):
    fig = plt.figure()
    y = predictors.apply(stand_detr,axis=0).values
    for i in range(len(predictors.keys())):
        plt.plot(y[:,i])
    plt.plot(stand_detr(target))
    return fig
    