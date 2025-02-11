import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class PreProcessNC:
    def __init__(self, file_pattern, var_name=None):
        """
        Initialize the analysis with a NetCDF file or pattern.
        
        Args:
            file_pattern (str): Path or pattern to your NetCDF file(s).
            var_name (str, optional): Name of the variable to analyze. 
                                      If not provided and only one variable exists,
                                      that variable will be used.
        """
        self.ds = xr.open_mfdataset(file_pattern, combine='by_coords')
        if var_name is None:
            data_vars = list(self.ds.data_vars.keys())
            if len(data_vars) == 1:
                self.var_name = data_vars[0]
            else:
                raise ValueError(
                    "Multiple variables found. Please specify one using the 'var_name' argument."
                )
        else:
            self.var_name = var_name

    def subset_time(self, start_date, end_date):
        """Subset the dataset by a given time range."""
        self.ds = self.ds.sel(time=slice(start_date, end_date))
        return self

    def extract_season(self, months):
        """
        Extract data for a specific season defined by month(s).
        
        Args:
            months (int or list of ints): Month(s) to extract (e.g., 6 or [6, 7, 8]).
        
        Returns:
            xarray.Dataset: Subset for the specified months.
        """
        if isinstance(months, int):
            months = [months]
        ds_season = self.ds.where(self.ds.time.dt.month.isin(months), drop=True)
        return ds_season

    def deseasonalize(self, ds_in, var_name=None, groupby="time.dayofyear"):
        """
        Remove the seasonal cycle by subtracting the climatology.
        """
        if var_name is None:
            var_name = self.var_name
        clim = ds_in[var_name].groupby(groupby).mean("time")
        anom = ds_in[var_name].groupby(groupby) - clim
        ds_out = ds_in.copy()
        ds_out[var_name] = anom
        return ds_out

    def detrend_linear(self, ds_in, var_name=None):
        """
        Remove the linear trend from the data.
        """
        if var_name is None:
            var_name = self.var_name
        poly = ds_in[var_name].polyfit(dim="time", deg=1)
        trend = xr.polyval(ds_in["time"], poly[var_name + "_polyfit_coefficients"])
        ds_out = ds_in.copy()
        ds_out[var_name] = ds_in[var_name] - trend
        return ds_out

    def subset_data(self, year=None, year_range=None, months=None, days=None):
        """
        Subset the dataset based on time criteria.
        
        Args:
            year (int, optional): A single year to filter on.
            year_range (tuple, optional): A tuple (start_year, end_year) to filter on.
            months (int or list of ints, optional): Month(s) to filter on.
            days (int or list of ints, optional): Day(s) of the month to filter on.
        
        Returns:
            xarray.Dataset: The filtered dataset.
        """
        ds_subset = self.ds
        
        # Filter by year or year range (but not both)
        if year is not None and year_range is not None:
            raise ValueError("Provide either a specific year or a year_range, not both.")
        if year is not None:
            ds_subset = ds_subset.sel(time=ds_subset.time.dt.year == year)
        if year_range is not None:
            start_year, end_year = year_range
            ds_subset = ds_subset.where(
                (ds_subset.time.dt.year >= start_year) & (ds_subset.time.dt.year <= end_year), 
                drop=True
            )
        # Filter by months if provided
        if months is not None:
            if isinstance(months, int):
                months = [months]
            ds_subset = ds_subset.where(ds_subset.time.dt.month.isin(months), drop=True)
        # Filter by days if provided
        if days is not None:
            if isinstance(days, int):
                days = [days]
            ds_subset = ds_subset.where(ds_subset.time.dt.day.isin(days), drop=True)
        return ds_subset

    def plot_field(self, var_name=None, extent=None, title=None, 
                   year=None, year_range=None, months=None, days=None, aggregate="mean"):
        """
        Plot a field based on various time filters.
        """
        if var_name is None:
            var_name = self.var_name
        
        ds_to_plot = self.subset_data(year=year, year_range=year_range, months=months, days=days)
        if ds_to_plot.time.size == 0:
            raise ValueError("No data found for the given filters.")
        
        if aggregate == "mean":
            field = ds_to_plot[var_name].mean(dim="time")
        elif aggregate == "sum":
            field = ds_to_plot[var_name].sum(dim="time")
        elif aggregate == "none":
            field = ds_to_plot[var_name]
        else:
            raise ValueError("Aggregate must be 'mean', 'sum', or 'none'.")
        
        filters = []
        if year is not None:
            filters.append(f"{year}")
        if year_range is not None:
            filters.append(f"{year_range[0]}-{year_range[1]}")
        if months is not None:
            filters.append(f"months={months}")
        if days is not None:
            filters.append(f"days={days}")
        default_title = f"{' '.join(filters)} - {var_name} ({aggregate})" if aggregate != "none" else f"{' '.join(filters)} - {var_name} (all time steps)"
        plot_title = title or default_title
        
        fig, ax = plt.subplots(figsize=(8, 5),
                               subplot_kw=dict(projection=ccrs.PlateCarree()))
        field.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            robust=True,
            cbar_kwargs={'label': plot_title}
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        plt.title(plot_title)
        plt.show()

    def compute_weighted_time_series(self, region_extent, groupby="time.dayofyear", lat_name="lat", lon_name="lon"):
        """
        Compute a weighted spatial average time series over a specified region.
        
        The weights are based on the cosine of the latitude (in radians) to account for grid cell area.
        
        Args:
            region_extent (list or tuple): [lon_min, lon_max, lat_min, lat_max] defining the region.
            groupby (str, optional): Grouping for deseasonalization. Default is "time.dayofyear".
            lat_name (str, optional): Name of the latitude coordinate. Default is "lat".
            lon_name (str, optional): Name of the longitude coordinate. Default is "lon".
        
        Returns:
            dict: Dictionary with keys "raw", "deseasonalized", and "detrended" containing the time series.
        """
        lon_min, lon_max, lat_min, lat_max = region_extent
        ds_region = self.ds.where(
            (self.ds[lon_name] >= lon_min) & (self.ds[lon_name] <= lon_max) &
            (self.ds[lat_name] >= lat_min) & (self.ds[lat_name] <= lat_max),
            drop=True
        )
        # Compute weights based on latitude (in radians)
        weights = np.cos(np.deg2rad(ds_region[lat_name]))
        # Broadcast weights to match the variable dimensions (assumes dims: time, lat, lon)
        weights, _ = xr.broadcast(weights, ds_region[self.var_name])
        
        # Raw weighted time series
        raw_ts = (ds_region[self.var_name] * weights).sum(dim=[lat_name, lon_name]) / weights.sum(dim=[lat_name, lon_name])
        
        # Deseasonalized
        ds_deseason = self.deseasonalize(ds_region, var_name=self.var_name, groupby=groupby)
        deseason_ts = (ds_deseason[self.var_name] * weights).sum(dim=[lat_name, lon_name]) / weights.sum(dim=[lat_name, lon_name])
        
        # Detrended (applied to the deseasonalized data)
        ds_detrend = self.detrend_linear(ds_deseason, var_name=self.var_name)
        detrend_ts = (ds_detrend[self.var_name] * weights).sum(dim=[lat_name, lon_name]) / weights.sum(dim=[lat_name, lon_name])
        
        return {"raw": raw_ts, "deseasonalized": deseason_ts, "detrended": detrend_ts}

    def plot_weighted_time_series(self, region_extent, groupby="time.dayofyear", lat_name="lat", lon_name="lon"):
        """
        Plot the weighted spatially averaged time series over a specific region.
        
        This plots the raw, deseasonalized, and detrended time series on one figure.
        
        Args:
            region_extent (list or tuple): [lon_min, lon_max, lat_min, lat_max] defining the region.
            groupby (str, optional): Grouping for deseasonalization. Default is "time.dayofyear".
            lat_name (str, optional): Name of the latitude coordinate. Default is "lat".
            lon_name (str, optional): Name of the longitude coordinate. Default is "lon".
        """
        ts_dict = self.compute_weighted_time_series(region_extent, groupby, lat_name, lon_name)
        
        plt.figure(figsize=(12, 6))
        ts_dict["raw"].plot(label="Raw", color="blue")
        ts_dict["deseasonalized"].plot(label="Deseasonalized", color="orange")
        ts_dict["detrended"].plot(label="Detrended", color="green")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(self.var_name)
        plt.title(f"Weighted Spatial Average Time Series\nRegion: {region_extent}")
        plt.show()
