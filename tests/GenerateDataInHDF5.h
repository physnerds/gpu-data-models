#ifndef GENERATEDATAINHDF5_H
#define GENERATEDATAINHDF5_H

#include <vector>
#include <string>
#include <fstream>
#include "HDFCxx.h"
#include "hdf5.h"


class GenerateDataInHDF5{
  public:
    GenerateDataInHDF5(std::string const &fname, int &chunksize);
    GenerateDataInHDF5(GenerateDataInHDF5&&) = default;
    GenerateDataInHDF5(GenerateDataInHDF5 const&)=default;
    ~GenerateDataInHDF5();
    
    struct DuneRawDataHeader{
      uint32_t chan_;
      float pedestal_;
      float sigma_;
      uint32_t Nadc_;
      int compression_;
      DuneRawDataHeader(uint32_t chan, float pedestal,float sigma, uint32_t Nadc, int compression):    chan_(chan),pedestal_(pedestal),sigma_(sigma),Nadc_(Nadc),compression_(compression){};
    };
    void CreateHeader(hid_t g,int channel_id,DuneRawDataHeader header_info);
    void WriteData(hid_t gid,std::string dset_name, std::vector<int>const &data);
    std::vector<int>ReturnFakeData(size_t dsize);
    
    
    private:
      hdf5::File file_;
      int chunksize_;
      
    
    
};

#endif