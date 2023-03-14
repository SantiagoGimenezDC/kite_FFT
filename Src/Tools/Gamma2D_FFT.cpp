/***********************************************************/
/*                                                         */
/*   Copyright (C) 2018-2022, M. Andelkovic, L. Covaci,    */
/*  A. Ferreira, S. M. Joao, J. V. Lopes, T. G. Rappoport  */
/*                                                         */
/***********************************************************/



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

#include <fstream>
#include "complex_op.hpp"


void integration(Eigen::Matrix<double, -1, 1 >& E_points, Eigen::Matrix<double, -1, 1 >& integrand, Eigen::Matrix<double, -1, 1 >& data){

  int E =E_points.size();

  //#pragma omp parallel for 
  for(int k=0; k<E; k++ ){ 
    if(abs(E_points(k))<0.99)
    for(int j=k; j<E; j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      double ej  = E_points(j),
	ej1      = E_points(j+1),
	de       = ej-ej1,
        integ    = ( integrand(j+1) + integrand(j) ) / 2.0;     
      if(abs(E_points(j))>0.99)
	    integ=0;
      data(k) -=  de * integ;
    }
    else
      data(k) =0;
      
  }
}


template <typename T,unsigned D>
void Simulation<T,D>::Gamma2D_FFT(int NRandomV, int NDisorder, int num_reps, std::vector<int> N_moments, int kernel, 
                              std::vector<std::vector<unsigned>> indices, std::string name_dataset){
  //  Sized must be divisible by num_reps!!!!
  int MEMORY_alt = N_moments.at(0);  
  int M = N_moments.at(0),  BUFFER_SIZE = r.Sized/num_reps; 
  

  
  // This function calculates all kinds of two-dimensional gamma matrices such
  // as Tr[V^a Tn v^b Tm] = G_nm
  //
  // The matrices are stored as
  //
  // | G_00   G_01   G_02   ...   G_0M | 
  // | G_10   G_11   G_12   ...   G_1M | 
  // | G_20   G_21   G_22   ...   G_2M | 
  // | ...    ...    ...    ...   ...  |
  // | G_N0   G_N1   G_N2   ...   G_NM | 
  //
  // For example, a 3x3 matrix would be represented as
  //
  // | G_00   G_01   G_02 | 
  // | G_10   G_11   G_12 | 
  // | G_20   G_21   G_22 | 
  //
  // This function calculates all the kinds of one-dimensional Gamma matrices
  // such as Tr[Tn]    Tr[v^xx Tn]     etc

  //  typedef typename extract_value_type<T>::value_type value_type;

  int num_velocities = 0;
  for(int i = 0; i < int(indices.size()); i++)
    num_velocities += indices.at(i).size();
  int factor = 1 - (num_velocities % 2)*2;

  //  --------- INITIALIZATIONS --------------
    
  KPM_Vector<T,D> kpm0(1, *this);      // initial random vector
  KPM_Vector<T,D> kpm1(3, *this); // vector that will be Chebyshev-iterated on
  KPM_Vector<T,D> kpm2(1, *this); // kpm1 multiplied by the velocity
  

  Eigen::Matrix<T, -1, -1>
     bras(BUFFER_SIZE,M),
     kets(BUFFER_SIZE,M); 

  bras.setZero();
  kets.setZero();
  
      


#pragma omp master
  {
    Global.FFT_cond.resize(M,1);
    Global.FFT_cond.setZero();
    Global.FFT_R_data.resize(M,NRandomV*NDisorder);  
  }
  
  Eigen::Matrix<double, -1, 1>
    integrand = Eigen::Matrix<double, -1, 1 >::Zero(M, 1),
    data      = Eigen::Matrix<double, -1, 1 >::Zero(M, 1),
    E_points  = Eigen::Matrix<double, -1, 1 >::Zero(M, 1);


  for (int e=0;e<M;e++)
    E_points(e) = cos(M_PI*(e+0.5)/M);
      
  // finished initializations


    
    
  // start the kpm iteration
  long average = 0;
  for(int disorder = 0; disorder < NDisorder; disorder++){
    h.generate_disorder();

    for(unsigned it = 0; it < indices.size(); it++)
      h.build_velocity(indices.at(it), it);

    for(int randV = 0; randV < NRandomV; randV++){

      integrand.setZero();
      h.generate_twists(); // Generates Random or fixed boundaries

      
      kpm0.initiate_vector();			// original random vector. This sets the index to zero
	
        kpm0.initiate_phases();           //Initiates the Hopping Phases, including TBC
	kpm1.initiate_phases();           //Initiates the Hopping Phases, including TBC
        kpm0.Exchange_Boundaries();
	
	
     for(int s=0; s<=num_reps; s++){
       
	int buffer_length = BUFFER_SIZE;
	
	if( s == num_reps ){
	  if(r.Sized % num_reps ==0)
	    break;
	  else{
	    buffer_length =  r.Sized - (s+1)*BUFFER_SIZE;
	    bras.resize(buffer_length, M);
            kets.resize(buffer_length, M);
	    bras.setZero();
            kets.setZero();
	  }
	}



	
        kpm1.v.setZero();
        kpm1.set_index(0);
        kpm0.set_index(0);
	kpm0.Velocity(&kpm1, indices, 0);	
       	
       	
	for(int i = 0; i <  M; i++) {
          kpm1.cheb_iteration(i);
              
          kpm2.set_index(0);
	  kpm1.Velocity(&kpm2, indices, 1);
          kpm2.empty_ghosts(0);

	  kets.col(i) = kpm2.v.matrix().col(0).segment(s*BUFFER_SIZE, buffer_length);
        }



       // copy the |0> vector to |kpm1>
	kpm1.v.setZero();
	kpm1.set_index(0);
        kpm1.v.col(0) = kpm0.v.col(0);
	

       // iterate M times, just like before. No need to multiply by v here	
        for(int i = 0; i < M; i++){
          kpm1.cheb_iteration(i);
	  bras.col(i) = kpm1.v.matrix().col(kpm1.index).segment(s*BUFFER_SIZE, buffer_length);
        }
	
	Bastin_FFTs( bras, kets, E_points, kernel, integrand); 
       


       integrand *= factor;
      }

     /*----------------------------------------------------------------------------------------------/
     
      #pragma omp critical
     {
       Global.FFT_cond +=  ( integrand - Global.FFT_cond ) / value_type(average + 1);
     }
     
     #pragma omp barrier
     {
      #pragma omp master
       {
	 T den = -1.0;//atoi(system.num_orbitals)*atoi(system.spin_degeneracy)/atoi(system.unit_cell_area)/units; 
  
         integrand = Global.FFT_cond;
         integration( E_points, integrand, data);	  

         std::ofstream dataP;
         dataP.open("currentResult_test.dat");

         for(int e=0;e<M;e++)  
           dataP<< E_points(e) <<"  "<< std::complex<double>(data(e)*den).real() <<std::endl;

          dataP.close();
       }
     }
      
     /----------------------------------------------------------------------------------------------*/

     
     store_data_FFT_runtime(&integrand, randV*NDisorder+disorder, name_dataset );
     average++;
     
    }	    
  }
  //store_data_FFT(&integrand, NDisorder*NRandomV-1, name_dataset );
}


template <typename T,unsigned D>
void Simulation<T,D>::store_data_FFT_runtime( Eigen::Matrix<double, -1, 1> *integrand, int RD,  std::string name_dataset){
  debug_message("Entered store_data_FFTx\n");

  typedef typename extract_value_type<T>::value_type value_type;
  
#pragma omp barrier
#pragma omp critical
  Global.FFT_cond.matrix() += ( *integrand - Global.FFT_cond.matrix() ) / value_type(RD + 1);
#pragma omp barrier
  
    
    
#pragma omp master
  {
    Eigen::Array <double, -1, -1> array_integrand((*integrand).rows(), 1);
    array_integrand.matrix() = *integrand;
    H5::H5File * file = new H5::H5File(name, H5F_ACC_RDWR);
    std::string R_label = name_dataset+"_RD"+std::to_string(RD+1);
    write_hdf5(array_integrand, file, R_label);
    write_hdf5(Global.FFT_cond, file, name_dataset);
    delete file;
  }
#pragma omp barrier    

    
  debug_message("Left store_data_FFT_runtime\n");
}


template <typename T,unsigned D>
void Simulation<T,D>::store_data_FFT( Eigen::Matrix<double, -1, 1> *integrand, int RD,  std::string name_dataset){
  debug_message("Entered store_data_FFTx\n");

  typedef typename extract_value_type<T>::value_type value_type;
  
#pragma omp barrier
#pragma omp critical
  Global.FFT_R_data.matrix().col(RD) += *integrand;
  Global.FFT_cond.matrix() += ( *integrand - Global.FFT_cond.matrix() ) / value_type(RD + 1);
#pragma omp barrier
  
    

    
#pragma omp master
  {    
    H5::H5File * file = new H5::H5File(name, H5F_ACC_RDWR);
    write_hdf5(Global.FFT_cond, file, name_dataset);
    write_hdf5(Global.FFT_R_data, file, name_dataset+"_R_data");
    delete file;
  }
#pragma omp barrier    

    
  debug_message("Left store_data_FFT\n");
}
 


#define instantiate(type, dim)  template void Simulation<type,dim>::Gamma2D_FFT(int, int, int, std::vector<int>, int, std::vector<std::vector<unsigned>>, std::string); \
  template void Simulation<type,dim>::store_data_FFT( Eigen::Matrix<double, -1, 1> *, int,  std::string  ); \
  template void Simulation<type,dim>::store_data_FFT_runtime( Eigen::Matrix<double, -1, 1> *, int,  std::string  );  

#include "instantiate.hpp"



