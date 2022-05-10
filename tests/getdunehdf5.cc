#include "GenerateDataInHDF5.h"
#include "HDFCxx.h"
#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <cstddef>

int main(int argc, char* argv[]){
    int chunk_size = 6000;
    auto _fdune = GenerateDataInHDF5("dune_file.h5",chunk_size);
    auto file = _fdune.GetFileObject();
    auto g_id = hdf5::Group::create(file,"APA01");
    int channel_id = 2500;
    int chan_ = 2500;
    float pedestal_ = 2305;
    float sigma_ = 8.32;
    int Nadc_ = 60;
    int compression_ = 0;
    
    auto header_info = GenerateDataInHDF5::DuneRawDataHeader(channel_id, pedestal_, 
                                                sigma_,  Nadc_, compression_);
   /* 
    GenerateDataInHDF5::DuneRawDataHeader header_info;
    
    header_info.chan_ = channel_id;
    header_info.sigma_ = sigma_;
    header_info.pedestal_ = pedestal_;
    header_info.Nadc_ = Nadc_;
    header_info.compression_ = compression_;
    */
    auto offset_channel = offsetof(GenerateDataInHDF5::DuneRawDataHeader,chan_);
    auto offset_pedestal = offsetof(GenerateDataInHDF5::DuneRawDataHeader,pedestal_);
    auto offset_sigma = offsetof(GenerateDataInHDF5::DuneRawDataHeader,sigma_);
    auto offset_nadc = offsetof(GenerateDataInHDF5::DuneRawDataHeader,Nadc_);
    auto offset_compression = offsetof(GenerateDataInHDF5::DuneRawDataHeader,compression_);
    
    std::cout<<"Offset Info "<<"\n"
        <<" channel "<<offset_channel<<"\n"
        <<" pedestal "<<offset_pedestal<<"\n"
        <<" sigma "<<offset_sigma<<"\n"
        <<" nadc "<<offset_nadc<<"\n"
        <<" sizeof hvl "<<sizeof(hvl_t)<<"\n"
        <<" size of header "<<sizeof(GenerateDataInHDF5::DuneRawDataHeader)<<"\n"
        <<std::endl;
   
    _fdune.CreateHeader(g_id,channel_id,header_info);
    
    auto fake_data = _fdune.ReturnFakeData(Nadc_);
    
    _fdune.WriteData(g_id,channel_id ,fake_data);
    
     g_id.Close();
    auto  obj_ct = H5Fget_obj_count(file,H5F_OBJ_GROUP);
    std::cout<<"Total open objects "<<obj_ct<<std::endl;
   
    
}