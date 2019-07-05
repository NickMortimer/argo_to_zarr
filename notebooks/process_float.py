import xarray as xr
import os
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
from distributed import Client
import dask
import zarr
from time import sleep
from tqdm import tqdm_notebook
import numcodecs
# import dask_mpi 
# from dask_mpi import initialize
# initialize()

from ncar_jobqueue import NCARCluster
cluster = NCARCluster(project="NIOW0001", cores=1, processes=1, memory="30GB", job_name='dask-cluster')
cluster.scale(10)
sleep(20)
client = Client(cluster)
print(cluster)


zarr_path =  '../../argo_float/argo.zarr'
dffiles = pd.read_csv('/glade/scratch/mortimer/argo/catalogue.csv')
files = list(dffiles.files)

from functools import partial

data_types ={'CONFIG_MISSION_NUMBER':'float32','CYCLE_NUMBER':'float32','DATA_CENTRE':'|U2','DATA_MODE':'|U1',
             'DATA_STATE_INDICATOR':'|U4','DC_REFERENCE':'|U32','DIRECTION':'|U1','FIRMWARE_VERSION':'|U32',
             'FLOAT_SERIAL_NO':'|U32','JULD':'float32','JULD_LOCATION':'float32','JULD_QC':'|U1','LATITUDE':'float32',
             'LONGITUDE':'float32','PI_NAME':'|U64','PLATFORM_NUMBER':'|U8','PLATFORM_TYPE':'|U32','POSITIONING_SYSTEM':'|U8',
             'POSITION_QC':'|U1','PRES':'float32','PRES_ADJUSTED':'float32','PRES_ADJUSTED_ERROR':'float32',
             'PRES_ADJUSTED_QC':'|U1','PRES_QC':'|U1','PROFILE_PRES_QC':'|U1','PROFILE_PSAL_QC':'|U1','PROFILE_TEMP_QC':'|U1',
             'PROJECT_NAME':'|U64','PSAL':'float32','PSAL_ADJUSTED':'float32','PSAL_ADJUSTED_ERROR':'float32',
             'PSAL_ADJUSTED_QC':'|U1','PSAL_QC':'|U1','TEMP':'float32','TEMP_ADJUSTED':'float32','TEMP_ADJUSTED_ERROR':'float32',
             'TEMP_ADJUSTED_QC':'|U1','TEMP_QC':'|U1','VERTICAL_SAMPLING_SCHEME':'|U256','WMO_INST_TYPE':'|U4'}

data_levels =['PRES','PRES_ADJUSTED','PRES_ADJUSTED_ERROR','PRES_ADJUSTED_QC','PRES_QC','PSAL','PSAL_ADJUSTED',
              'PSAL_ADJUSTED_ERROR','PSAL_ADJUSTED_QC','PSAL_QC','TEMP','TEMP_ADJUSTED','TEMP_ADJUSTED_ERROR',
              'TEMP_ADJUSTED_QC','TEMP_QC']

def process_mf(dsinput,levels,data_types=data_types,data_levels=data_levels):
    ds = xr.Dataset()
    #pad =xr.DataArray(np.ones((levels-len( dsinput.N_LEVELS),len(dsinput.N_PROF))) *np.nan,dims={'N_LEVELS','N_PROF'})
    #pad_qc = xr.DataArray(np.chararray((levels-len( dsinput.N_LEVELS),len(dsinput.N_PROF))),dims={'N_LEVELS','N_PROF'})
    dims =('N_PROF','N_LEVELS')
    pading =xr.DataArray(np.ones((len(dsinput.N_PROF),levels-len( dsinput.N_LEVELS))) *np.nan,dims=dims)
    pad_qc = xr.DataArray(np.chararray((len(dsinput.N_PROF),levels-len( dsinput.N_LEVELS))),dims=dims)
    pad_qc[:] = b' '
    for varname in data_types.keys():
        if varname in dsinput.data_vars:
            da = dsinput[varname]
            if 'N_LEVELS' in da.dims:   
                if varname in dsinput.data_vars:
                    if varname.endswith('QC'):
                        da = xr.concat([dsinput[varname],pad_qc],dim='N_LEVELS').astype(data_types[varname])
                    else:
                        da = xr.concat([dsinput[varname],pading],dim='N_LEVELS').astype(data_types[varname])
            else:
                da = dsinput[varname].astype(data_types[varname])
        else:
            if varname in data_levels:
                if data_types[varname]=='float32':
                    da = xr.DataArray(np.ones((len(dsinput.N_PROF),levels), dtype='float32')*np.nan , name=varname, dims=['N_PROF','N_LEVELS'])
                else:
                    p=np.chararray((len(dsinput.N_PROF),levels))
                    p[:]=b'0'
                    da = xr.DataArray(p.astype(data_types[varname]), name=varname, dims=['N_PROF','N_LEVELS'])
            else:
                if data_types[varname]=='float32':
                    da = xr.DataArray(np.ones(len(dsinput.N_PROF), dtype="float32")*np.nan , name=varname, dims=['N_PROF'])
                else:
                    p=np.chararray((len(dsinput.N_PROF)))
                    p[:]=b'0'
                    da = xr.DataArray(p.astype(data_types[varname]), name=varname, dims=['N_PROF'])
        if not ('HISTORY' in varname) and ('N_CALIB' not in da.dims) and ('N_PARAM' not in da.dims) and  ('N_PROF' in da.dims):
                ds[varname]= da
    return ds.chunk({'N_LEVELS':3000})
   
preproc = partial(process_mf,levels=3000)

@dask.delayed
def process_float(file):
    data = preproc(xr.open_dataset(file, chunks={'N_LEVELS':100}))
    return data




d =[]

start = 0
incr = 1000
stop = len(files)
ranges = list(range(start, stop, incr))
for i in  tqdm_notebook(ranges):
    print(f'Processing {i}')
    d = []
    for file in files[i:i+incr]:
        print(file)
        d.append(process_float(file))

    results = dask.compute(*d)

    t = xr.concat(results,dim='N_PROF', coords='minimal')    
    t = t.chunk({'N_PROF':10000,'N_LEVELS':3000})
    print(f'Finished concatenating dataset')
    
    numcodecs.blosc.use_threads = False
    synchronizer = zarr.ProcessSynchronizer('../../argozarr/argodask2.sync')
    #compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    zarr_path =  '../../argozarr/argo_dask2.zarr'
    #encoding = {vname: {'compressor': compressor} for vname in t.variables}
    d = t.to_zarr(zarr_path,mode='a',synchronizer=synchronizer,compute=True,append_dim='N_PROF')
    print('Appending Done!')
    client.restart()
    
    
cluster.close()
    
