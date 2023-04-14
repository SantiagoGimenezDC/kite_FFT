#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<Eigen/Dense>
#include<fftw3.h>




#include "Generic.hpp"
#include "ComplexTraits.hpp"
#include "myHDF5.hpp"
#include "Global.hpp"
#include "Random.hpp"
#include "Coordinates.hpp"
#include "LatticeStructure.hpp"
template <typename T, unsigned D>
class Hamiltonian;
template <typename T, unsigned D>
class KPM_Vector;
#include "Simulation.hpp"
#include "Hamiltonian.hpp"
#include "KPM_VectorBasis.hpp"
#include "KPM_Vector.hpp"




typedef double r_type;

r_type kernel_k(const int m, const int M){
      return 1/(M+1.0) *
           (
   	     (M-m+1.0) *
             cos( M_PI * m / (M + 1.0)) +
             sin( M_PI * m / (M + 1.0)) /
             tan( M_PI     / (M + 1.0))
	   );
    };




template <typename T,unsigned D>
void Simulation<T,D>::Greenwood_FFTs(   Eigen::Matrix<T, -1, -1>&  bras, Eigen::Matrix<T, -1, -1>& kets, Eigen::Matrix<double, -1, 1 >& E_points, int include_kernel, Eigen::Matrix<double, -1, 1 >& r_data){
  
  const std::complex<double> im(0,1);

  
  int M = bras.cols(),
    size = bras.rows();
  int l_start=0, l_end=size;
  
 
  r_type *kern  = new r_type [M],
    *IM_root      = new r_type[M],
    *thread_data  = new r_type[M];

  
  for(int m=0;m<M;m++){
    kern[m]      =  1.0;
    IM_root[m]     =  sin( acos( E_points[m] )  );
  }
  
  if(include_kernel == 1)
    for(int m=0;m<M;m++)
      kern[m] = kernel_k(m,M);

    fftw_plan plan1, plan2, plan3, plan4;

    double
      *bra_re = ( double* ) fftw_malloc(sizeof(double) * M ),
      *bra_im = ( double* ) fftw_malloc(sizeof(double) * M ),
      *ket_re = ( double* ) fftw_malloc(sizeof(double) * M ),
      *ket_im = ( double* ) fftw_malloc(sizeof(double) * M );
    
        
# pragma omp critical
    {
      plan1 = fftw_plan_r2r_1d(M, bra_re, bra_re, FFTW_REDFT01, FFTW_ESTIMATE);//bra_re
      plan2 = fftw_plan_r2r_1d(M, bra_im, bra_im, FFTW_REDFT01, FFTW_ESTIMATE); //bra_im
      
      plan3 = fftw_plan_r2r_1d(M, ket_re, ket_re, FFTW_REDFT01, FFTW_ESTIMATE);//bra_re
      plan4 = fftw_plan_r2r_1d(M, ket_im, ket_im, FFTW_REDFT01, FFTW_ESTIMATE); //bra_im
    }

    
    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	std::complex<r_type> tmp = std::complex<r_type>(bras(l,m));
	
	bra_re[m] =  kern[m] * tmp.real();
        bra_im[m] =  kern[m] * tmp.imag();

	tmp = std::complex<r_type>(kets(l,m));
	
	ket_re[m] =  kern[m] * tmp.real();
        ket_im[m] =  kern[m] * tmp.imag(); 
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   

      fftw_execute(plan3);
      fftw_execute(plan4);   
      



    for(int j=0; j<M; j++ )
      thread_data[j] += (  (bra_re[j] - im * bra_im[j] )   *   ( ket_re[j] + im *  ket_im[j] )  ).real();
    
    
    }

    for(int e=0;e<M;e++)
      r_data(e) += 2.0 * thread_data[e] / ( IM_root[e] * IM_root[e] );

    # pragma omp critical
    {
      

      fftw_destroy_plan(plan1);
      fftw_free(bra_re);
      fftw_destroy_plan(plan2);
      fftw_free(bra_im);

      
      fftw_destroy_plan(plan3);
      fftw_free(ket_re);
      fftw_destroy_plan(plan4);
      fftw_free(ket_im);
    }

  delete []thread_data;    
  delete []IM_root;
  delete []kern;
}


#define instantiate(type, dim)  template void Simulation<type,dim>::Greenwood_FFTs(Eigen::Matrix<type, -1, -1>&, Eigen::Matrix<type, -1, -1>& , Eigen::Matrix<double, -1, 1 >& , int,Eigen::Matrix<double, -1, 1 >& );

#include "instantiate.hpp"
