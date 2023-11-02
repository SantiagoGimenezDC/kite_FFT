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




//Just the Jackson kernel. Same as in postprocessing;
double kernel(const int m, const int M){
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

  
  std::complex<double> factors [M];// Factors that preceed the FFTs operations
  double IM_root[M]; //Factors needed AFTER the FFTs operations;

  
  for(int m=0;m<M;m++){
    factors[m] = (2.0-(m==0)) * std::polar(1.0,M_PI*m/(2.0*M));
    IM_root[m] = sqrt(1.0-E_points(m)*E_points(m));
  }

  if(include_kernel == 1)
    for(int m=0;m<M;m++)
      factors[m] *= kernel(m,M);


  /*================================================================================================================================================//
  //   FFTW variables; The real and imaginary parts of the random vector need to be processed separately, as if they were different random vectors; //
  //   The reason for that is, FFTW provides the Fourier transform as e^(2*i*pi*m*k). We need the transform proportional to e^(i*pi*m*k),           //
  //   without the factor 2 within the exponent. Hence, a change of variable k=2j is needed (Weisse/2006), and such change of variables             //
  //   requires conjugating the  FFT output for odd entries. Conjugation would act over the imaginary part of the rand vec too if done naively,     //
  //   giving wrong results. Hence, to preserve the imaginary part of the randVec, it is necessary to treat it separetely. At  least thats the      //
  //   only way I found of doing it.                                                                                                                //
  //================================================================================================================================================*/
  
  
  //8 plans + 14 [M] sized vectors per thread. Can it be reduced to 2 plans and some 4 vectors?
    
  fftw_plan plan1, plan2, plan3, plan4,
              plan5, plan6, plan7, plan8;

  fftw_complex
    //Green's functions. The real part of this output yields the third necessary Fourier transform, which is the Delta function
    *bra_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //real part of rand vec;
    *bra_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //imag part of rand vec;
    //Derivative Green's function output
    *bra_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //real part of rand vec;
    *bra_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //imag part of rand vec;

    //Green's functions. The real part of this output yields the third necessary Fourier transform, which is the Delta function
    *ket_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //real part of rand vec;
    *ket_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //imag part of rand vec;
    //Derivative Green's function output
    *ket_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ), //real part of rand vec;
    *ket_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ); //imag part of rand vec;


  std::complex<double> bra, ket, //variables to access values within bras/kets         
                         p[M], w[M]; //Dot product partial results;

  for(int k=0;k<M;k++){
    p[k] = 0;
    w[k] = 0;
  }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
  {
    plan1 = fftw_plan_dft_1d(M, bra_re,   bra_re,    FFTW_BACKWARD, FFTW_ESTIMATE); 
    plan2 = fftw_plan_dft_1d(M, bra_im,   bra_im,    FFTW_BACKWARD, FFTW_ESTIMATE); 
    plan3 = fftw_plan_dft_1d(M, bra_D_re, bra_D_re,  FFTW_BACKWARD, FFTW_ESTIMATE); 
    plan4 = fftw_plan_dft_1d(M, bra_D_im, bra_D_im,  FFTW_BACKWARD, FFTW_ESTIMATE); 

    plan5 = fftw_plan_dft_1d(M, ket_re,   ket_re,    FFTW_BACKWARD, FFTW_ESTIMATE); 
    plan6 = fftw_plan_dft_1d(M, ket_im,   ket_im,    FFTW_BACKWARD, FFTW_ESTIMATE); 
    plan7 = fftw_plan_dft_1d(M, ket_D_re, ket_D_re,  FFTW_BACKWARD, FFTW_ESTIMATE); 
    plan8 = fftw_plan_dft_1d(M, ket_D_im, ket_D_im,  FFTW_BACKWARD, FFTW_ESTIMATE); 
  }

  for(int l=0; l<size;l++){
    for(int m=0;m<M;m++){
	//The following casting allows to simplify all template instantiations: converts all to double.
	//All operations here are performed in double precision.
	//I also don`t think fftw supports long double as of right now.
	//The row-wise accessed of the col-major matrices are some of the longest operations in this loop!!
	bra = std::complex<double>(bras(l,m)); 

	//Bra Greens functions FFT inputs:
	bra_re[m][0] = ( factors[m] ).real() * bra.real(); 
	bra_re[m][1] = ( factors[m] ).imag() * bra.real(); 


	bra_im[m][0] = ( factors[m] ).real() * bra.imag(); 
	bra_im[m][1] = ( factors[m] ).imag() * bra.imag(); 

	
	//The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
	bra_D_re[m][0] = m * bra_re[m][0]; 
	bra_D_re[m][1] = m * bra_re[m][1]; 


	bra_D_im[m][0] = m * bra_im[m][0]; 
	bra_D_im[m][1] = m * bra_im[m][1]; 




	
	ket = std::complex<double>(kets(l,m)); 	

	//Ket Greens functions FFT inputs:	
	ket_re[m][0] = ( factors[m] ).real() * ket.real(); 
	ket_re[m][1] = ( factors[m] ).imag() * ket.real(); 


	ket_im[m][0] = ( factors[m] ).real() * ket.imag(); 
	ket_im[m][1] = ( factors[m] ).imag() * ket.imag(); 

	
	//The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
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
      
      //As mentioned previously, variable change as k=2j; This loop updates the partial results of the dot products p(E) and w(E);
      //This also performs the conjugation from bra.cdot(ket) of the complex random vector dot product.
      for(int j=0; j<M/2; j++){
	//Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).  
        p[2*j] += ( bra_re[j][0] - im * bra_im[j][0] ) *  //Re(G(k)) *
	          ( ket_re  [j][0] + im * ket_re  [j][1]  +  im * ( ket_im  [j][0] + im * ket_im  [j][1] ) ) + //G(k)+
	  
	          ( bra_re[j][0] - im * bra_re  [j][1] - im * ( bra_im  [j][0] - im * bra_im  [j][1] ) ) * //G(k)
	          ( ket_re[j][0] + im * ket_im[j][0] );   //Re(G(k))

	

	//Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
	w[2*j] += ( bra_D_re[j][0] - im * bra_D_re[j][1] - im * ( bra_D_im[j][0] - im * bra_D_im[j][1] ) )    * //dG(k)
	          ( ket_re[j][0] + im * ket_im[j][0] ) - //Re(G(k))
	  
	          ( bra_re[j][0] - im * bra_im[j][0] )     *  //Re(G(k))
	          ( ket_D_re[j][0] + im * ket_D_re[j][1] + im * ( ket_D_im[j][0] + im * ket_D_im[j][1] ) ); //dG(k)



	//Odd parts of p(k) and w(k), involving the conjugation of the second half of the outputs of the FFTs:
	p[2*j+1] += ( bra_re[M-j-1][0] - im * bra_im[M-j-1][0] )    *
	            ( ket_re[M-j-1][0] - im * ket_re  [M-j-1][1] + im * ( ket_im  [M-j-1][0] - im * ket_im  [M-j-1][1] ) ) +
	  
	            ( bra_re[M-j-1][0] + im * bra_re  [M-j-1][1] - im * ( bra_im  [M-j-1][0] + im * bra_im  [M-j-1][1] ) ) *
	            ( ket_re[M-j-1][0] + im * ket_im[M-j-1][0] );


	w[2*j+1] += ( bra_D_re[M-j-1][0] + im * bra_D_re[M-j-1][1] - im * ( bra_D_im[M-j-1][0] + im * bra_D_im[M-j-1][1] ) ) *
	            ( ket_re[M-j-1][0] + im * ket_im[M-j-1][0] ) -
	   
	            ( bra_re[M-j-1][0] - im * bra_im[M-j-1][0] ) *
	            ( ket_D_re[M-j-1][0] - im * ket_D_re[M-j-1][1] + im * ( ket_D_im[M-j-1][0] - im * ket_D_im[M-j-1][1] ) );
      }      
  }

    //Keeping just the real part of E*p(E)+im*sqrt(1-E^2)*w(E) yields the Kubo-Bastin integrand:
  for(int k=0;k<M;k++){
    double ek  = E_points[k], new_value;
    new_value = ek * ( p[k] ).real() + ( im * IM_root[k] * w[k] ).real();
    new_value *= 1.0/pow( (1.0 - ek  * ek ), 2.0);
    new_value *= -1.0/(M_PI); //Matches the prefactors from fill.cpp;
                                      //From the paper this would be -4.0/(M_PI*M_PI);
    final_integrand[k] += new_value;
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
    } 
}



#define instantiate(type, dim)  template void Simulation<type,dim>::Bastin_FFTs(Eigen::Matrix<type, -1, -1>&, Eigen::Matrix<type, -1, -1>& , Eigen::Matrix<double, -1, 1 >& , int,Eigen::Matrix<double, -1, 1 >& );

#include "instantiate.hpp"
