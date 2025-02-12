import xarray as xr
xr.set_options(display_style="text")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


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
        # If the provided file_pattern is not an absolute path, assume it's relative to data/raw.
        if not os.path.isabs(file_pattern):
            # Determine the base path. Assuming this module is in src/, the project root is one level up.
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
            file_pattern = os.path.join(base_dir, file_pattern)
            print("Resolved file pattern:", file_pattern)

        self.ds = xr.open_mfdataset(file_pattern, combine='by_coords')
        
        # Rename coordinates if they exist
        rename_dict = {}
        if "valid_time" in self.ds.coords:
            rename_dict["valid_time"] = "time"
        if "latitude" in self.ds.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in self.ds.coords:
            rename_dict["longitude"] = "lon"

        if rename_dict:
            self.ds = self.ds.rename(rename_dict)
            print(f"Renamed coordinates: {rename_dict}")
  
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
        return self

    def deseasonalize(self, ds_in=None, var_name=None, groupby="time.dayofyear"):
        if ds_in is None:
            ds_in = self.ds
            update_self = True
        else:
            update_self = False
        if var_name is None:
            var_name = self.var_name
        clim = ds_in[var_name].groupby(groupby).mean("time")
        anom = ds_in[var_name].groupby(groupby) - clim
        ds_out = ds_in.copy()
        ds_out[var_name] = anom
        if update_self:
            self.ds = ds_out
            return self
        else:
            return ds_out
    
    def detrend_linear(self, ds_in=None, var_name=None):
        if ds_in is None:
            ds_in = self.ds
            update_self = True
        else:
            update_self = False
        if var_name is None:
            var_name = self.var_name
        poly = ds_in[var_name].polyfit(dim="time", deg=1)
        coeff_key = var_name + "_polyfit_coefficients"
        if coeff_key not in poly:
            coeff_key = "polyfit_coefficients"
        trend = xr.polyval(ds_in["time"], poly[coeff_key])
        ds_out = ds_in.copy()
        ds_out[var_name] = ds_in[var_name] - trend
        if update_self:
            self.ds = ds_out
            return self
        else:
            return ds_out

    def subset_data(self, ds_in=None, year=None, year_range=None, months=None, days=None, 
                    region_extent=None, lat_name="lat", lon_name="lon"):
        """
        Subset the dataset based on time and optionally spatial criteria,
        and adjust the longitude coordinates (projection) if needed.
        
        Args:
            ds_in (xarray.Dataset, optional): The dataset to filter. If None, self.ds is used.
            year (int, optional): A single year to filter on.
            year_range (tuple, optional): A tuple (start_year, end_year) to filter on.
            months (int or list of ints, optional): Month(s) to filter on.
            days (int or list of ints, optional): Day(s) of the month to filter on.
            region_extent (list or tuple, optional): [lon_min, lon_max, lat_min, lat_max] defining the spatial region.
            lat_name (str, optional): Name of the latitude coordinate. Default is "lat".
            lon_name (str, optional): Name of the longitude coordinate. Default is "lon".
            
        Returns:
            xarray.Dataset: A new dataset filtered according to the criteria.
        """
        # Use ds_in if provided, otherwise use self.ds
        if ds_in is None:
            ds_in = self.ds
        ds_subset = ds_in.copy()

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
        
        # Adjust longitude projection if needed (i.e., if lon values are > 180)
        if ds_subset[lon_name].max() > 180:
            ds_subset = ds_subset.assign_coords(
                lon=(((ds_subset[lon_name] + 180) % 360) - 180)
            ).sortby(lon_name)
        
        # Apply spatial filtering if region_extent is provided
        if region_extent is not None:
            lon_min, lon_max, lat_min, lat_max = region_extent
            ds_subset = ds_subset.where(
                (ds_subset[lon_name] >= lon_min) & (ds_subset[lon_name] <= lon_max) &
                (ds_subset[lat_name] >= lat_min) & (ds_subset[lat_name] <= lat_max),
                drop=True
            )
        
        return ds_subset

    def plot_spatial_field(self, region_extent=None, var_name=None, title=None, aggregate="mean",
                        lat_name="lat", lon_name="lon", year=None, months=None, days=None, ds_in=None):
        """
        Plot a spatial field map, optionally restricted to a specific region and time filters.
        
        When aggregate is "none", a separate panel is created for each time step with a single,
        shared colorbar and a global title that includes the region extent, year, variable name, and month(s).
        
        Args:
            region_extent (list or tuple, optional): [lon_min, lon_max, lat_min, lat_max] defining the region.
                If None, the full dataset is used.
            var_name (str, optional): Variable to plot. Defaults to the instance variable.
            title (str, optional): Overall title for the plot. If not provided, a default title is generated.
            aggregate (str, optional): How to aggregate over time ('mean', 'sum', or 'none').
            lat_name (str, optional): Name of the latitude coordinate. Default is "lat".
            lon_name (str, optional): Name of the longitude coordinate. Default is "lon".
            year (int, optional): A specific year to filter on.
            months (int or list of ints, optional): Month(s) to filter on.
            days (int or list of ints, optional): Day(s) of the month to filter on.
            ds_in (xarray.Dataset, optional): A dataset to use for plotting instead of self.ds.
        
        Returns:
            None. Displays the plot.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if var_name is None:
            var_name = self.var_name

        # Filter the dataset by time using the subset_data method.
        ds_time_filtered = self.subset_data(ds_in=ds_in, year=year, months=months, days=days)

        # Remap longitudes if needed (from 0-360 to -180-180)
        ds_to_plot = ds_time_filtered
        if ds_to_plot[lon_name].max() > 180:
            ds_to_plot = ds_to_plot.assign_coords(
                lon=(((ds_to_plot[lon_name] + 180) % 360) - 180)
            ).sortby(lon_name)

        # Subset by region if provided.
        if region_extent is not None:
            ds_to_plot = ds_to_plot.where(
                (ds_to_plot[lon_name] >= region_extent[0]) &
                (ds_to_plot[lon_name] <= region_extent[1]) &
                (ds_to_plot[lat_name] >= region_extent[2]) &
                (ds_to_plot[lat_name] <= region_extent[3]),
                drop=True
            )
        
        # Build a global title that includes the key information.
        default_title = (f"Spatial Field Map: {var_name} | Region: "
                        f"{region_extent if region_extent is not None else 'Full Domain'} | "
                        f"Year: {year} | Month(s): {months}")
        global_title = title or default_title

        if aggregate in ["mean", "sum"]:
            # For aggregated cases, compute the aggregation and make a single-panel plot.
            if aggregate == "mean":
                field = ds_to_plot[var_name].mean(dim="time")
            else:
                field = ds_to_plot[var_name].sum(dim="time")

            fig, ax = plt.subplots(1, 1, figsize=(8, 5),
                                subplot_kw=dict(projection=ccrs.PlateCarree()))
            field.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                robust=True,
                cbar_kwargs={'label': var_name}
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            if region_extent is not None:
                ax.set_extent(region_extent, crs=ccrs.PlateCarree())
            ax.set_title(global_title)
            plt.show()

        elif aggregate == "none":
            # For "none", plot each time slice individually with a shared colorbar.
            field = ds_to_plot[var_name]
            if "time" in field.dims:
                num_panels = field.sizes["time"]
                # Compute global color limits so all panels share the same scale.
                vmin = float(field.min().values)
                vmax = float(field.max().values)

                # Create subplots in a single row (adjust figsize as needed).
                fig, axes = plt.subplots(1, num_panels, figsize=(8 * num_panels, 5),
                                        subplot_kw=dict(projection=ccrs.PlateCarree()))
                # Ensure axes is always iterable.
                if num_panels == 1:
                    axes = [axes]
                im = None  # To store the mappable from the first plot.
                for i, ax in enumerate(axes):
                    sub_field = field.isel(time=i)
                    # Format the date string for the subplot title.
                    date_str = np.datetime_as_string(sub_field["time"].values, unit='D')
                    im = sub_field.plot(
                        ax=ax,
                        transform=ccrs.PlateCarree(),
                        cmap="RdBu_r",
                        robust=True,
                        add_colorbar=False,  # Turn off individual colorbars.
                        vmin=vmin,
                        vmax=vmax
                    )
                    ax.coastlines()
                    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax.add_feature(cfeature.LAND, facecolor="lightgray")
                    if region_extent is not None:
                        ax.set_extent(region_extent, crs=ccrs.PlateCarree())
                    ax.set_title(date_str)
                # Adjust the layout to leave space for the vertical colorbar and the global title.
                fig.subplots_adjust(right=0.85, top=0.85)
                # Create a single vertical colorbar.
                cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
                cbar.set_label(var_name)
                plt.suptitle(global_title, fontsize=16)
                plt.show()
            else:
                # If there's no time dimension, revert to a single-panel plot.
                fig, ax = plt.subplots(1, 1, figsize=(8, 5),
                                    subplot_kw=dict(projection=ccrs.PlateCarree()))
                field.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap="RdBu_r",
                    robust=True,
                    cbar_kwargs={'label': var_name}
                )
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor="lightgray")
                if region_extent is not None:
                    ax.set_extent(region_extent, crs=ccrs.PlateCarree())
                ax.set_title(global_title)
                plt.show()
        else:
            raise ValueError("Aggregate must be 'mean', 'sum', or 'none'.")





    def compute_weighted_time_series(self, region_extent, groupby="time.dayofyear", 
                                    lat_name="lat", lon_name="lon", 
                                    year=None, months=None, days=None, ds_in=None):
        """
        Compute a weighted spatial average time series over a specified region,
        with optional time filtering.
        
        The weights are based on the cosine of the latitude (in radians) to account for grid cell area.
        
        Args:
            region_extent (list or tuple): [lon_min, lon_max, lat_min, lat_max] defining the region.
            groupby (str, optional): Grouping for deseasonalization. Default is "time.dayofyear".
            lat_name (str, optional): Name of the latitude coordinate. Default is "lat".
            lon_name (str, optional): Name of the longitude coordinate. Default is "lon".
            year (int, optional): A specific year to filter on.
            months (int or list of ints, optional): Month(s) to filter on.
            days (int or list of ints, optional): Day(s) of the month to filter on.
            ds_in (xarray.Dataset, optional): A dataset to use for computing the time series.
        
        Returns:
            dict: Dictionary with keys "raw", "deseasonalized", and "detrended" containing the time series.
        """
        # Use ds_in if provided; otherwise, use self.ds filtered by time.
        ds_time = self.subset_data(ds_in=ds_in, year=year, months=months, days=days)

        # Spatial subsetting:
        lon_min, lon_max, lat_min, lat_max = region_extent
        ds_region = ds_time.where(
            (ds_time[lon_name] >= lon_min) & (ds_time[lon_name] <= lon_max) &
            (ds_time[lat_name] >= lat_min) & (ds_time[lat_name] <= lat_max),
            drop=True
        )

        # Compute weights based on latitude (in radians)
        weights = np.cos(np.deg2rad(ds_region[lat_name]))
        weights, _ = xr.broadcast(weights, ds_region[self.var_name])

        # Raw weighted time series:
        raw_ts = (ds_region[self.var_name] * weights).sum(dim=[lat_name, lon_name]) / weights.sum(dim=[lat_name, lon_name])

        # Deseasonalized dataset:
        ds_deseason = self.deseasonalize(ds_in=ds_region, var_name=self.var_name, groupby=groupby)
        deseason_ts = (ds_deseason[self.var_name] * weights).sum(dim=[lat_name, lon_name]) / weights.sum(dim=[lat_name, lon_name])

        # Detrended dataset:
        ds_detrend = self.detrend_linear(ds_in=ds_deseason, var_name=self.var_name)
        detrend_ts = (ds_detrend[self.var_name] * weights).sum(dim=[lat_name, lon_name]) / weights.sum(dim=[lat_name, lon_name])

        return {"raw": raw_ts, "deseasonalized": deseason_ts, "detrended": detrend_ts}

    def plot_weighted_time_series(self, region_extent, groupby="time.dayofyear", 
                                lat_name="lat", lon_name="lon", 
                                year=None, months=None, days=None, ds_in=None):
        """
        Plot the weighted spatially averaged time series over a specific region,
        with optional time filtering.
        
        This plots the raw, deseasonalized, and detrended time series in separate panels.
        
        Args:
            region_extent (list or tuple): [lon_min, lon_max, lat_min, lat_max] defining the region.
            groupby (str, optional): Grouping for deseasonalization. Default is "time.dayofyear".
            lat_name (str, optional): Name of the latitude coordinate. Default is "lat".
            lon_name (str, optional): Name of the longitude coordinate. Default is "lon".
            year (int, optional): A specific year to filter on.
            months (int or list of ints, optional): Month(s) to filter on.
            days (int or list of ints, optional): Day(s) of the month to filter on.
            ds_in (xarray.Dataset, optional): A dataset to use for computing the time series.
        """
        ts_dict = self.compute_weighted_time_series(region_extent, groupby, lat_name, lon_name,
                                                    year=year, months=months, days=days, ds_in=ds_in)
        
        fig, axes = plt.subplots(nrows=4, figsize=(12, 16), sharex=True)

        # Plot each time series in a separate panel
        ts_dict["raw"].plot(ax=axes[0], color="blue")
        axes[0].set_title("Raw Time Series")
        axes[0].set_ylabel(self.var_name)

        ts_dict["deseasonalized"].plot(ax=axes[1], color="orange")
        axes[1].set_title("Deseasonalized Time Series")
        axes[1].set_ylabel(self.var_name)

        ts_dict["detrended"].plot(ax=axes[2], color="green")
        axes[2].set_title("Detrended Time Series")
        axes[2].set_ylabel(self.var_name)
            
        # Compute and plot the trend (Raw - Detrended)
        trend = ts_dict["deseasonalized"] - ts_dict["detrended"]
        trend.plot(ax=axes[3], color="red")
        axes[3].set_title("Trend")
        axes[3].set_ylabel(self.var_name)

        # Common x-axis label
        axes[3].set_xlabel("Time")

        plt.suptitle(f"Weighted Spatial Average Time Series\nRegion: {region_extent}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit suptitle
        plt.show()
