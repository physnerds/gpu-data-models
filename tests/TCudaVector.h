#ifndef CUDAVECTORHEADERDEF
#define CUDAVECTORHEADERDEF

#include <cmath>
#include <iostream>
#include <cassert>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


template <typename T>
class CudaVector
{
private:
   thrust::host_vector<T>h_vec;
   thrust::device_vector<T>d_vec;

   int mSize;  // size of vector

public:


   //constructors
   CudaVector(std::vector<T>& otherVector);
   CudaVector(int size);
   
   //destructors
   ~CudaVector(){};
   //overloading some operators....
   CudaVector<T>& operator=(const CudaVector& otherVector);
   T& operator[](int i);  // see element
   
   //common functions...
   
   std::vector<T> CopyHostVector();
   std::vector<T>CopyDeviceVector();
   
   thrust::device_vector<T> ReturnDeviceVector();
   
   thrust::device_ptr<T> WrapRawPtr(T* raw_ptr);
   T* GetDeviceArrayPointer(thrust::device_vector<T>temp_vec);
   
   int Getsize();
   
   
};

template<typename T>
CudaVector<T>::CudaVector(std::vector<T>& otherVector){
    h_vec = otherVector;
    mSize = otherVector.size();
}

template<typename T>
CudaVector<T>::CudaVector(int i){
    h_vec.resize(i); //create a host vector with storage for i elements;
    mSize = i;
}

template<typename T>
CudaVector<T>& CudaVector<T>::operator=(const CudaVector& otherVector)
{
   mSize == otherVector.mSize;
   h_vec = otherVector.h_vec;
   return *this;
}

template<typename T>
T& CudaVector<T>::operator[](int i){
    assert(i>-1);
    assert(i<mSize);
    return h_vec[i];    
}

template<typename T>
std::vector<T> CudaVector<T>::CopyHostVector(){
    assert(mSize >0);
    std::vector<T>temp_vec(mSize);
    thrust::copy(h_vec.begin(), h_vec.end(), temp_vec.begin());
    return temp_vec;
    
}

template<typename T>
std::vector<T> CudaVector<T>::CopyDeviceVector(){
    assert(d_vec.size()!=0);
    std::vector<T>temp_vec(d_vec.size());
    thrust::copy(d_vec.begin(),d_vec.end(),temp_vec.begin());
 return temp_vec;   
}

template<typename T>
thrust::device_vector<T> CudaVector<T>:: ReturnDeviceVector(){
    assert(mSize>0);
    thrust::device_vector<T> temp_vec(h_vec);
    
    return temp_vec;
    
}

template<typename T>
T* CudaVector<T>::GetDeviceArrayPointer(thrust::device_vector<T>temp_vec){
    return thrust::raw_pointer_cast(&temp_vec[0]);
       
}


//pointer that can access the device memory
template<typename T>
thrust::device_ptr<T>CudaVector<T>::WrapRawPtr(T* raw_ptr){
    cudaMalloc((void**)&raw_ptr,mSize*sizeof(T));
    thrust::device_ptr<T>dev_ptr(raw_ptr);
    thrust::fill(dev_ptr,dev_ptr+mSize,(T)0);
    return dev_ptr;
    
}

template<typename T>
int CudaVector<T>::Getsize(){
        return mSize;
}
#endif