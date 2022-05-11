#include "OffloadIntoHost.h"
#include "ReadRawDataHDF5.h"
#include "HDFCxx.h"
#include "hdf5.h"
#include <iostream>
#include <cstring>

OffloadIntoHost::OffloadIntoHost(std::string f_name):file_(hdf5::File::open(f_name.c_str())){
    
}

OffloadIntoHost::OffloadIntoHost(){

}

OffloadIntoHost::~OffloadIntoHost(){

}

hdf5::Group OffloadIntoHost::GetAPAHandle(std::string apa_name){
    hdf5::Group apa_id = file_.GetAPAHandle(apa_name);
    return apa_id;
    
}

CudaVector  OffloadIntoHost::OffloadHeaderIntoCudaArray(ReadRawDataHDF5::DuneRawDataHeader header){
    std::vector<ReadRawDataHDF5::DuneDataHeader>temp_vec = {header};
    return CudaVector<ReadRawDataHDF5::DuneDataHeader>(temp_vec);
    
}

template<typename T>
CudaVector OffloadIntoHost::OffloadDataIntoCudaArray(std::vector<T> data){
    
    return CudaVector<T>(data);
    
}

template<typename T>
void OffloadIntoHost::CopyData(T* dest_data, T* src_data, unsigned int data_size){
    size_t size = data_size*sizeof(T);
    memcpy(dest_data,src_data,size);
    
}
