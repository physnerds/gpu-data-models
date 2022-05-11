#IFNDEF OFFLOADINTOHOST_H
#DEFINE OFFLOADINTOHOST_H

#include <vector>
#include <string>
#include <fstream>
#include "HDFCxx.h"
#include "hdf5.h"
#include "ReadRawDataHDF5.h"

#include "TCudaVector.h"
#include <helper_cuda.h>

class OffloadIntoHost{
  public:
    //Constructor
    OffloadIntoHost(std:string f_name);
    OffloadIntoHost();
    //Destructor
    ~OffloadIntoHost();
    
    hdf5::Group GetAPAHandle(std::string apa_name);
    CudaVector OffloadHeaderIntoCudaArray(ReadRawDataHDF5::DuneRawDataHeader header);
    
    template<typename T>
    CudaVector OffloadDataIntoCudaArray(std::vector<T> data);
    
    template<typename T>
    void CopyData(T* dest_data, T* src_data, unsigned int data_dim);
    private:
      hdf5::File file_;
      
    
};

#endif