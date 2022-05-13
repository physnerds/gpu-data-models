#include "OffloadIntoHost.h"
#include "ReadRawDataHDF5.h"
#include "HDFCxx.h"
#include "hdf5.h"
#include <iostream>
#include <cstring>

OffloadIntoHost::OffloadIntoHost(std::string f_name):file_(ReadRawDataHDF5(f_name.c_str())){
    
}

OffloadIntoHost::~OffloadIntoHost(){}

hdf5::Group OffloadIntoHost::GetAPAHandle(std::string apa_name){
    hdf5::Group apa_id = file_.GetAPAHandle(apa_name);
    return apa_id;
    
}

CudaVector<ReadRawDataHDF5::DuneRawDataHeader>  OffloadIntoHost::OffloadHeaderIntoCudaArray(hdf5::Group apa_id, int ch_id){
    ReadRawDataHDF5::DuneRawDataHeader c_container[1];
    file_.GetChannelHeader(apa_id, ch_id,c_container); 
    std::vector<ReadRawDataHDF5::DuneRawDataHeader>vec = {c_container[0]};
    CudaVector<ReadRawDataHDF5::DuneRawDataHeader>wrapper(vec);
    return wrapper;
}

template<typename T>
CudaVector<T> OffloadIntoHost::OffloadDataIntoCudaArray(std::vector<T> data){
    
    return CudaVector<T>(data);
    
}

template<typename T>
void OffloadIntoHost::CopyData(T* dest_data, T* src_data, unsigned int data_size){
    size_t size = data_size*sizeof(T);
    memcpy(dest_data,src_data,size);
    
}
