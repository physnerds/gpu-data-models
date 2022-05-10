#include "ReadRawDataHDF5.h"
#include "HDFCxx.h"
#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <cstddef>

int main(int argc, char* argv[]){
    
    std::string f_name = "dune_file.h5";
    std::string apa_name = "APA01";
    int ch_num = 2500;
    auto f_handle = ReadRawDataHDF5(f_name);
    
    ReadRawDataHDF5::DuneRawDataHeader c_data[1];
    f_handle.GetChannelHeader(apa_name,ch_num,c_data);
    std::cout<<"Chan "<<c_data[0].chan_<<"\n"
        <<"Pedestal "<<c_data[0].pedestal_<<"\n"
        <<"Sigma "<<c_data[0].sigma_<<"\n"
        <<" NAdc "<<c_data[0].Nadc_<<"\n"
        <<std::endl;
    
    //okay now directly read the data....
    
    hdf5::Group apa_id = f_handle.GetAPAHandle("APA01");
    auto container = f_handle.GetChannelData( apa_id, c_data[0].chan_ ,c_data[0].Nadc_);
    for(auto val:container)std::cout<<val<<std::endl;
}