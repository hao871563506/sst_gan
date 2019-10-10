import fsspec
import xarray as xr 

# just get the first 10 timesteps for now
# should change 10 to 53 in actual training
uris = [f'gcs://pangeo-ocean-ml/LLC4320/SST.{tstep:010d}.zarr'
        for tstep in range(0, 4088+1, 73)][:10]

dsets = [xr.open_zarr(fsspec.get_mapper(uri), consolidated=True)
         for uri in uris]

ds = xr.combine_nested(dsets, 'timestep')
print(ds)

# drop nan data first
# because nan is the same for every timestep, so we only need
# to use ds.SST[0] to calculate nan value for all timestep 
num_nans = ds.SST[0].isnull().sum(dim=['x', 'y']).load()
sst_valid = ds.SST.where(num_nans == 0, drop=True)
print(sst_valid)

# coarse all the data by a factor of 32
sst_coarse = sst_valid.coarsen(x=32, y=32).mean()
print(sst_coarse)

# all data is stored in ds.SST, should use load() to explicitly load the image
# for timestep in range(10):
# 	for region in range(312):
# 		hr_image = sst_valid[timestep, region].load()
#		lr_image = sst_coarse[timestep, region].load()

