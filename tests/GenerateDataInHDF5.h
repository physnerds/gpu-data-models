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
    GenerateDataInHDF5();
    ~GenerateDataInHDF5();
    
    struct DuneRawDataHeader{
      int chan_;
      float pedestal_;
      float sigma_;
      int Nadc_;
      int compression_;
      DuneRawDataHeader(int chan, float pedestal,float sigma, int Nadc, int compression):    chan_(chan),pedestal_(pedestal),sigma_(sigma),Nadc_(Nadc),compression_(compression){};
    };
    void CreateHeader(hid_t g,int channel_id,DuneRawDataHeader header_info);
    void WriteData(hid_t gid,int ch_id, std::vector<int>const &data);
    void CreateChannelGroup();
    std::vector<int>ReturnFakeData(size_t dsize);
    
    hdf5::File GetFileObject();
    private:
      hdf5::File file_;
      int chunksize_;
      
    
    
};

#endif