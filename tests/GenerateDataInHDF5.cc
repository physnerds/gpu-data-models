#include "HDFCxx.h"
#include "hdf5.h"
#include "GenerateDataInHDF5.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <set>
#include <time.h>
#include <random>

//start with the constructors I guess..
GenerateDataInHDF5::GenerateDataInHDF5(std::string const& fname, int &chunksize):
    file_(hdf5::File::create(fname.c_str())),chunksize_(chunksize){
    
    
}
//destructor
GenerateDataInHDF5::~GenerateDataInHDF5(){
    
}

void GenerateDataInHDF5::CreateHeader(hid_t g,int channel_id,DuneRawDataHeader header_info){
    constexpr hsize_t ndims = 1;
    constexpr hsize_t dims[ndims]={0};
    const hsize_t chunk_dims[ndims] = {static_cast<hsize_t>(chunksize_)};
    const hsize_t max_dims[ndims] = {header_info.Nadc_};
    
    auto space = hdf5::Dataspace::create_simple(ndims,dims,max_dims);
    auto prop = hdf5::Property::create();
    prop.set_chunk(ndims,chunk_dims);
    //hdf5::Group g = hdf5::Group::create(file_,"APA");
    
    auto cid = hdf5::CompoundData::create(sizeof(DuneRawDataHeader));
    
    cid.insert_datatype<uint32_t>("Chan",HOFFSET(DuneRawDataHeader,chan_));
    cid.insert_datatype<float>("Pedestal",HOFFSET(DuneRawDataHeader,pedestal_));
    cid.insert_datatype<float>("Sigma",HOFFSET(DuneRawDataHeader,sigma_));
    cid.insert_datatype<uint32_t>("nADC",HOFFSET(DuneRawDataHeader,Nadc_));
    cid.insert_datatype<int>("Compression",HOFFSET(DuneRawDataHeader,compression_));
    
    //now the data
    std::string header_name = "ChannelHeader_"+std::to_string(channel_id);
    std::string d_name = "ChannelID_"+std::to_string(channel_id);
    auto dset_h = hdf5::Dataset::create_cd(g,header_name.c_str(),
                                                   space,cid, H5P_DEFAULT);
    //write the header during data-set creation and get done with it.
    //dset_h.write_cd(space, cid, hid_t filespace,header_info);
    
    hdf5::Dataset dset_d = hdf5::Dataset::create<int>(g,d_name.c_str(),space,prop);
    
    
    //now need to create datasets for the channel header and channel data 
    //create 2 data-sets..one for header and one for actual APA data.
    
}
void GenerateDataInHDF5::WriteData(hid_t gid,std::string dset_name, std::vector<int>const &data){
    auto dset_id = hdf5::Dataset::open(gid,dset_name.c_str());
    hsize_t ndims = 1;
    auto dspace_h = hdf5::Dataspace::get_space(dset_id);
    hsize_t max_dims[ndims];
    hsize_t old_dims[ndims];
    
    H5Sget_simple_extent_dims(dspace_h,old_dims,max_dims);
    hsize_t new_dims[ndims] = {old_dims[0]+data.size()};
    hsize_t slab_size[ndims] = {data.size()};
    dset_id.set_extent(new_dims);
    auto new_space = hdf5::Dataspace::get_space(dset_id);
    new_space.select_hyperslab(old_dims,slab_size);
    auto mem_space = hdf5::Dataspace::create_simple(ndims,slab_size,max_dims);
    dset_id.write(mem_space,new_space,data);
        
}

std::vector<int> GenerateDataInHDF5::ReturnFakeData(size_t dlength){
    std::vector<int>fake_data;
    fake_data.reserve(dlength);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> some_rand(2000,3000);
    
    for(int i=0;i<dlength;i++)fake_data.push_back(some_rand(rng));
    
    return fake_data;
}