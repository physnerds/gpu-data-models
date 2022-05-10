#include "ReadRawDataHDF5.h"
#include "HDFCxx.h"
#include "hdf5.h"
#include <iostream>
#include <cstring>
//eventually I need to get rid of this by separating the data-struct from GenerateRawData class.


ReadRawDataHDF5::ReadRawDataHDF5(std::string file_name):file_(hdf5::File::open(file_name.c_str())){
    
}

ReadRawDataHDF5::~ReadRawDataHDF5(){ }


void ReadRawDataHDF5::GetChannelHeader(std::string apa_name,int ch_id, DuneRawDataHeader c_container[1]){
    auto cid = hdf5::CompoundData::create(sizeof(DuneRawDataHeader)); 
    cid.insert_datatype<int>("Chan",HOFFSET(DuneRawDataHeader,chan_));
    cid.insert_datatype<double>("Pedestal",HOFFSET(DuneRawDataHeader,pedestal_));
    cid.insert_datatype<double>("Sigma",HOFFSET(DuneRawDataHeader,sigma_));
    cid.insert_datatype<int>("nADC",HOFFSET(DuneRawDataHeader,Nadc_));
    cid.insert_datatype<int>("Compression",HOFFSET(DuneRawDataHeader,compression_));
    
    auto name = "ChannelHeader_"+std::to_string(ch_id);
    auto apa_id = hdf5::Group::open(file_,apa_name.c_str());
    
    std::cout<<"Opening the data-set "<<apa_id<<std::endl;
    auto ch_header = hdf5::Dataset::open(apa_id,name.c_str());
    
    ch_header.read_cd(cid, c_container);
    
}

std::vector<uint32_t> ReadRawDataHDF5::GetChannelData(hdf5::Group apa_id, int channel_id,int size){
    uint32_t container[size];
    std::vector<uint32_t> vec_container(size);
    auto name = "ChannelID_"+std::to_string(channel_id);
    auto ch_data = hdf5::Dataset::open(apa_id,name.c_str());
    
    ch_data.read(container,H5T_STD_I32LE);
    memcpy(&vec_container[0], &container[0], size*sizeof(uint32_t));
    return vec_container;
    
}

hdf5::Group ReadRawDataHDF5::GetAPAHandle(std::string apa_name){
    auto apa_id = hdf5::Group::open(file_,apa_name.c_str());
    
    return apa_id;
}