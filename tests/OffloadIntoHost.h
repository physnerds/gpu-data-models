#ifndef OFFLOADINTOHOST_H
#define OFFLOADINTOHOST_H

#include <vector>
#include <string>
#include <fstream>
#include "HDFCxx.h"
#include "hdf5.h"
#include "ReadRawDataHDF5.h"

#include "TCudaVector.h"
#include "helper_functions/helper_cuda.h"

class OffloadIntoHost{
  public:
    //Constructor
    OffloadIntoHost(std::string f_name);
    //Destructor
    ~OffloadIntoHost();
    
    hdf5::Group GetAPAHandle(std::string apa_name);
    CudaVector<ReadRawDataHDF5::DuneRawDataHeader>OffloadHeaderIntoCudaArray(hdf5::Group apa_id, int ch_id);
    
    template<typename T>
    CudaVector<T> OffloadDataIntoCudaArray(std::vector<T> data);
    
    template<typename T>
    void CopyData(T* dest_data, T* src_data, unsigned int data_dim);
    
    private:
      ReadRawDataHDF5 file_;
      
    
};

#endif