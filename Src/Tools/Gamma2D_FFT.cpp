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

//Just a time measurement verbose output 
void Station(int millisec, std::string msg ){
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<msg;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl;

}

template <typename T,unsigned D>
void Simulation<T,D>::Gamma2D_FFT(int NRandomV, int NDisorder, int num_reps, std::vector<int> N_moments, int kernel, 
                              std::vector<std::vector<unsigned>> indices, std::string name_dataset){
  
  int MEMORY_alt = N_moments.at(0);  
  int M = N_moments.at(0),
    BUFFER_SIZE = r.Size/num_reps; //Controls the row-wise size of the Chebyshev vector buffer
  

  
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
    
  KPM_Vector<T,D> kpm0(1, *this); // initial random vector
  KPM_Vector<T,D> kpm1(3, *this); // vector that will be Chebyshev-iterated on
  KPM_Vector<T,D> kpm2(1, *this); // kpm1 multiplied by the velocity

  //Hack to avoid through ghosts inside the buffer. This was relevant for the TBG.
  KPM_Vector<T,D> ghost_ref(1, *this);
  ghost_ref.v.col(0) = Eigen::Matrix<T, -1, 1 >::Constant( ghost_ref.v.col(0).size(),1, 1.0);
  ghost_ref.empty_ghosts(0);

  #pragma omp master
  {
    std::cout<<" Hilbert space dim:   t*size:"<< r.n_threads*r.Size<<",  orb:"<<r.Orb<<",   Nt: "<<r.Nt<<"  sizet: "<<r.Sizet <<std::endl;
  }

  //Declaration and initialization of the row-wise Chebyshev vector buffers needed for the FFTs. These are the largest memory blocks.
  //From  the paper, these are the a^{R/L} matrix row blocks;
  Eigen::Matrix<T, -1, -1>
     bras(BUFFER_SIZE,M),
     kets(BUFFER_SIZE,M); 

  bras.setZero();
  kets.setZero();

  
  //global FFT variables
  #pragma omp master
  {
    Global.FFT_cond.resize(M,1);
    Global.FFT_cond.setZero();
    Global.FFT_R_data.resize(M,NRandomV*NDisorder);  
  }

  //Local FFT variables
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
	


      //Variable that dictates current position along the Chebyshev vector rows
      //the FFT algorithm is operating on, skipping the ghosts;  
      int current = 0;
      while(std::complex<double>(ghost_ref.v.matrix().col(0)(current)).real() == 0)
        current++;

      
	
     for(int s=0; s<=num_reps; s++){

       auto start_RV = std::chrono::steady_clock::now();

       //-------------------This snippet allows sized to not necessarily be divisible by num_reps.------------// 
       int buffer_length = BUFFER_SIZE;
	if( s == num_reps ){
	  if( r.Size % num_reps == 0 )
	    break;
	  else{
	    buffer_length =  r.Size % BUFFER_SIZE;
	    bras.resize(buffer_length, M);
            kets.resize(buffer_length, M);
	    bras.setZero();
            kets.setZero();
	  }
	}
       //------------------------------------------------------------------------------------------------------//
    
       #pragma omp master
       {
	 std::cout<<" RD step: "<<disorder*NRandomV+randV+1<<"/"<<NDisorder*NRandomV<<"  - s: "<<s+1<<"/"<<num_reps+(r.Size % num_reps>0)<<std::endl;
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

	 //This would also be saving ghost values equal to 0 in kets:
	 //kets.col(i) = kpm2.v.matrix().col(0).segment(s*BUFFER_SIZE, buffer_length);

	  
	 //------------hack to avoid the ghosts
	 int j=0;
	 for(int n=0; j<buffer_length;n++)
	   if( std::complex<double>(ghost_ref.v.matrix().col(0)(current+n)).real() != 0 ){
	     kets(j,i)=kpm2.v(current+n,0);
	     j++;
	   }
     	 //----------------------------------------------------//	  
       }

       //copy the |0> vector to |kpm1>
       kpm1.v.setZero();
       kpm1.set_index(0);
       kpm1.v.col(0) = kpm0.v.col(0);
	

       //iterate M times, just like before. No need to multiply by v here	
       for(int i = 0; i < M; i++){
         kpm1.cheb_iteration(i);
	  
  	 //This would also be saving ghost values equal to 0 in bras:
	 //bras.col(i) = kpm1.v.matrix().col(kpm1.index).segment(s*BUFFER_SIZE, buffer_length);

	  
	 //---------------hack to avoid the ghosts-------------//
	 int j=0;
	 for(int n=0; j<buffer_length;n++)
	   if( std::complex<double>(ghost_ref.v.matrix().col(0)(current+n)).real() != 0 ){
             bras(j,i)=kpm1.v(current+n,kpm1.index);
	     j++;	
	   }
	 //----------------------------------------------------//
        }

	
	//--------------FINAL::hack to skip the ghosts---------//	
        int m=0;
	for(int j=0;j<buffer_length && current+m<r.Sized;){
	  if( std::complex<double>(ghost_ref.v.matrix().col(0)(current+m)).real() != 0 )
	    j++;
	    m++;
	}
	current+=m;

	while(std::complex<double>(ghost_ref.v.matrix().col(0)(current)).real() == 0 && current<r.Sized )
          current++;	 
	//----------------------------------------------------//


	

        //===============================PERFORMING FFTs+DOT PRODUCT ALONG THE STORED ROWS OF BRAS/KETS====================================// 
	auto start_FFT = std::chrono::steady_clock::now();    

	Bastin_FFTs( bras, kets, E_points, kernel, integrand); 
       
        auto end_FFT = std::chrono::steady_clock::now();    
        #pragma omp master
        {
	  Station(std::chrono::duration_cast<std::chrono::milliseconds>(end_FFT - start_FFT).count(), "       FFT Time:          "); 
        }

	integrand *= factor;
        //=================================================================================================================================// 


	
        auto end_RV = std::chrono::steady_clock::now();    
	
        #pragma omp master
        {
	  Station(std::chrono::duration_cast<std::chrono::milliseconds>(end_RV - start_RV).count(), "       Total partition step time:         "); 
          std::cout<<std::endl;       
        }
     }//End of num_reps cycle


     store_data_FFT_runtime(&integrand, randV*NDisorder+disorder, name_dataset );
     average++;

     #pragma omp master
     {
       std::cout<<std::endl<<std::endl;
     }

    
    }//End of randV cycle	    
  }//End of NDisorder cycle
  
}


template <typename T,unsigned D>
void Simulation<T,D>::store_data_FFT_runtime( Eigen::Matrix<double, -1, 1> *integrand, int RD,  std::string name_dataset){
  debug_message("Entered store_data_FFTx\n");

  typedef typename extract_value_type<T>::value_type value_type;
  
#pragma omp barrier
#pragma omp critical
  Global.FFT_R_data.matrix().col(RD) += *integrand;
#pragma omp barrier
  
    
    
#pragma omp master
  {
    
    Global.FFT_cond.matrix() += ( Global.FFT_R_data.matrix().col(RD) - Global.FFT_cond.matrix() ) / value_type(RD + 1);

    Eigen::Array <double, -1, -1> array_integrand( Global.FFT_R_data.matrix().rows(), 1);
    array_integrand.matrix() = Global.FFT_R_data.matrix().col(RD);
    
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



