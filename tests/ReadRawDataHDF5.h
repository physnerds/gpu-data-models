#ifndef READRAWDATAHDF5_H
#define READRAWDATAHDF5_H

#include <vector>
#include <string>
#include <fstream>
#include "HDFCxx.h"
#include "hdf5.h"

class ReadRawDataHDF5{
  public:
    
    struct DuneRawDataHeader{
      int chan_;
      float pedestal_;
      float sigma_;
      int Nadc_;
      int compression_;       
    };
    
    
    ReadRawDataHDF5(std::string file_name);
    ~ReadRawDataHDF5();
    void GetAPAInfo();
    hdf5::Group GetAPAHandle(std::string apa_number);
    void GetChannelHeader(std::string apa_name, int ch_id, DuneRawDataHeader dat_container[1]);
    size_t GetTotalChannelNumbers(hid_t apa_id);
    
    std::vector<uint32_t> GetChannelData(hdf5::Group apa_id, int channel_id,int size); 
    herr_t apa_info(hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata);
    
    
  private:
    hdf5::File file_;
    std::vector<std::string>apa_names;
};

#endif
