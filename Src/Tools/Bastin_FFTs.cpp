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

r_type kernel(const int m, const int M){
      return 1/(M+1.0) *
           (
   	     (M-m+1.0) *
             cos( M_PI * m / (M + 1.0)) +
             sin( M_PI * m / (M + 1.0)) /
             tan( M_PI     / (M + 1.0))
	   );
    };







template <typename T,unsigned D>
void Simulation<T,D>::Bastin_FFTs(   Eigen::Matrix<T, -1, -1>&  bras, Eigen::Matrix<T, -1, -1>& kets, Eigen::Matrix<double, -1, 1 >& E_points, int include_kernel, Eigen::Matrix<double, -1, 1 >& final_integrand){

  const std::complex<double> im(0,1);  


  int M = bras.cols(),
    size = bras.rows();

  std::complex<r_type> *factors = new std::complex<r_type> [M];
  r_type *IM_root = new r_type [M];

  
  for(int m=0;m<M;m++){
    factors[m] = (2.0-(m==0)) * std::polar(1.0,M_PI*m/(2.0*M));
    IM_root[m] = sin( acos(E_points(m)) );
  }

  if(include_kernel == 1)
    for(int m=0;m<M;m++)
      factors[m] *= kernel(m,M);

  
  int l_start=0, l_end=size;
  
    //8 planos+ 14 vetores [M] por thread. Pode ser reduzido a 2 planos e uns 4 vetores?
    
    fftw_plan plan1, plan2, plan3, plan4,
              plan5, plan6, plan7, plan8;

    std::complex<r_type>
      *bra_Green = new std::complex<r_type> [M],
      *bra_Delta = new std::complex<r_type> [M],
      *bra_Dfull = new std::complex<r_type> [M],
      
      *ket_Green = new std::complex<r_type> [M],
      *ket_Delta = new std::complex<r_type> [M],
      *ket_Dfull = new std::complex<r_type> [M];   

    
    fftw_complex   
      *bra_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *bra_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),

      *bra_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *bra_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),


      *ket_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *ket_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),

      *ket_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *ket_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );


    
    r_type *thread_integrand = new r_type [M];

    for(int m=0;m<M;m++)
      thread_integrand[m]=0;
    

    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra_re,   bra_re,    FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative
      plan2 = fftw_plan_dft_1d(M, bra_im,   bra_im,    FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative      
      plan3 = fftw_plan_dft_1d(M, bra_D_re, bra_D_re,  FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative
      plan4 = fftw_plan_dft_1d(M, bra_D_im, bra_D_im,  FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative

      plan5 = fftw_plan_dft_1d(M, ket_re,   ket_re,    FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan6 = fftw_plan_dft_1d(M, ket_im,   ket_im,    FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative      
      plan7 = fftw_plan_dft_1d(M, ket_D_re, ket_D_re,  FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan8 = fftw_plan_dft_1d(M, ket_D_im, ket_D_im,  FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){

	//The following casting allows to simplify all template instantiations. All operations here are performed in double precision anyways,
	// I don`t think fftw supports long double.
	
	bra_Green[m] = std::complex<r_type>(bras(l,m)); //This single row-wise access of a col-major matrix is the longest operation in this loop!!
	
	bra_re[m][0] = ( conj(factors[m]) ).real() * bra_Green[m].real(); 
	bra_re[m][1] = ( conj(factors[m]) ).imag() * bra_Green[m].real(); 


	bra_im[m][0] = ( conj(factors[m]) ).real() * bra_Green[m].imag(); 
	bra_im[m][1] = ( conj(factors[m]) ).imag() * bra_Green[m].imag(); 


	bra_D_re[m][0] = m * bra_re[m][0]; 
	bra_D_re[m][1] = m * bra_re[m][1]; 


	bra_D_im[m][0] = m * bra_im[m][0]; 
	bra_D_im[m][1] = m * bra_im[m][1]; 




	ket_Green[m] = std::complex<r_type>(kets(l,m)); 	

	ket_re[m][0] = ( factors[m] ).real() * ket_Green[m].real(); 
	ket_re[m][1] = ( factors[m] ).imag() * ket_Green[m].real(); 


	ket_im[m][0] = ( factors[m] ).real() * ket_Green[m].imag(); 
	ket_im[m][1] = ( factors[m] ).imag() * ket_Green[m].imag(); 


	ket_D_re[m][0] = m * ket_re[m][0]; 
	ket_D_re[m][1] = m * ket_re[m][1]; 


	ket_D_im[m][0] = m * ket_im[m][0]; 
	ket_D_im[m][1] = m * ket_im[m][1]; 
	
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   
      fftw_execute(plan3);
      fftw_execute(plan4);

      fftw_execute(plan5);
      fftw_execute(plan6);
      fftw_execute(plan7);
      fftw_execute(plan8);
      

      for(int j=0; j<M/2; j++){

	/*//This gives the Greenwood formula
	bras(l,2*j)= bra_re  [j][0] + im * bra_re  [j][1] - im * ( bra_im  [j][0] + im * bra_im  [j][1] );
	kets(l,2*j)=ket_Green[2*j] = ket_re  [j][0] + im * ket_re  [j][1] + im * ( ket_im  [j][0] + im * ket_im  [j][1] );
	bras(l,2*j+1) = bra_Green[2*j+1] = bra_re  [M-j-1][0] - im * bra_re  [M-j-1][1] - im * ( bra_im  [M-j-1][0] - im * bra_im  [M-j-1][1] );
	kets(l,2*j+1)=ket_Green[2*j+1] = ket_re  [M-j-1][0] - im * ket_re  [M-j-1][1] + im * ( ket_im  [M-j-1][0] - im * ket_im  [M-j-1][1] );*/



	

	
	bra_Delta[2*j] = bra_re  [j][0]                       - im *   bra_im  [j][0];	
	bra_Green[2*j] = bra_re  [j][0] + im * bra_re  [j][1] - im * ( bra_im  [j][0] + im * bra_im  [j][1] );//BRA is conjugated for the dot product
	bra_Dfull[2*j] = bra_D_re[j][0] + im * bra_D_re[j][1] - im * ( bra_D_im[j][0] + im * bra_D_im[j][1] );

	ket_Delta[2*j] = ket_re  [j][0]                       + im *   ket_im  [j][0];	
	ket_Green[2*j] = ket_re  [j][0] + im * ket_re  [j][1] + im * ( ket_im  [j][0] + im * ket_im  [j][1] );
	ket_Dfull[2*j] = ket_D_re[j][0] + im * ket_D_re[j][1] + im * ( ket_D_im[j][0] + im * ket_D_im[j][1] );


	
	bra_Delta[2*j+1] = bra_re  [M-j-1][0]                           - im *   bra_im  [M-j-1][0];	
	bra_Green[2*j+1] = bra_re  [M-j-1][0] - im * bra_re  [M-j-1][1] - im * ( bra_im  [M-j-1][0] - im * bra_im  [M-j-1][1] );//FFT algo conversion forces conjugation in re and im parts
	bra_Dfull[2*j+1] = bra_D_re[M-j-1][0] - im * bra_D_re[M-j-1][1] - im * ( bra_D_im[M-j-1][0] - im * bra_D_im[M-j-1][1] );

	ket_Delta[2*j+1] = ket_re  [M-j-1][0]                           + im * ( ket_im  [M-j-1][0]  );	
	ket_Green[2*j+1] = ket_re  [M-j-1][0] - im * ket_re  [M-j-1][1] + im * ( ket_im  [M-j-1][0] - im * ket_im  [M-j-1][1] );//FFT algo conversion forces conjugation in re and im parts
	ket_Dfull[2*j+1] = ket_D_re[M-j-1][0] - im * ket_D_re[M-j-1][1] + im * ( ket_D_im[M-j-1][0] - im * ket_D_im[M-j-1][1] );
      }
      




      for(int m=0; m<M; m++ )    
      thread_integrand[m] += (
			       bra_Delta[m] * ( E_points(m) * ket_Green[m] - im * IM_root[m] * ket_Dfull[m] ) +
			       ket_Delta[m] * ( E_points(m) * bra_Green[m] + im * IM_root[m] * bra_Dfull[m] )   
 			     ).real();
        

    }
    
      
  for(int m=0; m<M; m++ ){
    r_type ek  = E_points(m);
    final_integrand(m) +=  4 * thread_integrand[m] /(  M_PI * pow( (1.0 - ek  * ek ), 2.0 ) ) ; //am I missing a - sign somewhere???
  }

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

      fftw_destroy_plan(plan5);
      fftw_free(bra_D_re);
      fftw_destroy_plan(plan6);
      fftw_free(bra_D_im); 

      
      fftw_destroy_plan(plan7);
      fftw_free(ket_D_re);
      fftw_destroy_plan(plan8);
      fftw_free(ket_D_im); 

      delete []bra_Green;
      delete []bra_Delta;
      delete []bra_Dfull;
      
      delete []ket_Green;
      delete []ket_Delta;
      delete []ket_Dfull;
        
      delete []thread_integrand;
      delete []factors;
      delete []IM_root;

    } 
}




#define instantiate(type, dim)  template void Simulation<type,dim>::Bastin_FFTs(Eigen::Matrix<type, -1, -1>&, Eigen::Matrix<type, -1, -1>& , Eigen::Matrix<double, -1, 1 >& , int,Eigen::Matrix<double, -1, 1 >& );

#include "instantiate.hpp"
