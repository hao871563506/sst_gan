import fsspec
import xarray as xr 

# just get the first 10 timesteps for now
# uris = [f'../sst_datasets/LLC4320/SST.{tstep:010d}.zarr'
#         for tstep in range(0, 4088+1, 73)][:10]
uris = [f'gcs://pangeo-ocean-ml/LLC4320/SST.{tstep:010d}.zarr'
        for tstep in range(0, 4088+1, 73)][:10]


dsets = [xr.open_zarr(fsspec.get_mapper(uri), consolidated=True)
         for uri in uris]

ds = xr.combine_nested(dsets, 'timestep')
print(ds)