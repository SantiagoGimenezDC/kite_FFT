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


template <typename T,unsigned D>
void Simulation<T,D>::Gamma2D_mod(int NRandomV, int NDisorder, int NReps,std::vector<int> N_moments, 
                              std::vector<std::vector<unsigned>> indices, std::string name_dataset){
  //Static memory alocation is overriden; MEMORY_alt here is supposed to be large 
  //Eigen::Matrix<T, MEMORY, MEMORY> tmp; 

  int MEMORY_alt = N_moments.at(0)/NReps;
  Eigen::Matrix<T, -1, -1> tmp(MEMORY_alt, MEMORY_alt);
  
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

  typedef typename extract_value_type<T>::value_type value_type;

  int num_velocities = 0;
  for(int i = 0; i < int(indices.size()); i++)
    num_velocities += indices.at(i).size();
  int factor = 1 - (num_velocities % 2)*2;

  //  --------- INITIALIZATIONS --------------
    
  KPM_Vector<T,D> kpm0(1, *this);      // initial random vector
  KPM_Vector<T,D> kpm1(2, *this); // left vector that will be Chebyshev-iterated on
  KPM_Vector<T,D> kpm2(MEMORY_alt, *this); // right vector that will be Chebyshev-iterated on
  KPM_Vector<T,D> kpm3(MEMORY_alt, *this); // kpm1 multiplied by the velocity

  // initialize the local gamma matrix and set it to 0
  int size_gamma = 1;
  for(int i = 0; i < 2; i++){
    if(N_moments.at(i) % 2 != 0){
      std::cout << "The number of moments must be an even number, due to limitations of the program. Aborting\n";
      exit(1);
    }
    size_gamma *= N_moments.at(i);
  }

  //This buffer grows too large, proportional to ~M^2. For M>40000, memory cost is already >25.6GB/core (double complex):
  // Eigen::Array<T, -1, -1> gamma = Eigen::Array<T, -1, -1 >::Zero(1, size_gamma);   
  //Instead, we use a single global Global.general_gamma, and declare the extra R averaged variable Global.general_gamma_R: 
#pragma omp master
  {
    Global.general_gamma   = Eigen::Array<T, -1, -1 > :: Zero(N_moments.at(0), N_moments.at(1));
    Global.general_gamma_R = Eigen::Array<T, -1, -1 > :: Zero(N_moments.at(0), N_moments.at(1)); 
  }

  // finished initializations


  
  // start the kpm iteration
  long average = 0;
  for(int disorder = 0; disorder < NDisorder; disorder++){
    h.generate_disorder();
    for(unsigned it = 0; it < indices.size(); it++)
      h.build_velocity(indices.at(it), it);

    for(int randV = 0; randV < NRandomV; randV++){
	  h.generate_twists(); // Generates Random or fixed boundaries

      kpm0.initiate_vector();			// original random vector. This sets the index to zero

	  kpm0.initiate_phases();           //Initiates the Hopping Phases, including TBC
	  kpm1.initiate_phases();          
	  kpm2.initiate_phases();          
	  kpm3.initiate_phases();          
                                        
      kpm0.Exchange_Boundaries();
      kpm1.set_index(0);      
      kpm0.Velocity(&kpm1, indices, 0);
      
      // run through the left loop MEMORY_alt iterations at a time
      for(int n = 0; n < N_moments.at(0); n+=MEMORY_alt)
        {

	  // Iterate MEMORY_alt times. The first time this occurs, we must exclude the zeroth
          // case, because it is already calculated, it's the identity
          for(int i = n; i < n + MEMORY_alt; i++) {
              kpm1.cheb_iteration(i);
              
              kpm3.set_index(i%MEMORY_alt);
              kpm1.Velocity(&kpm3, indices, 1);
              kpm3.empty_ghosts(i%MEMORY_alt);
            }
	  
          // copy the |0> vector to |kpm2>
          kpm2.set_index(0);
          kpm2.v.col(0) = kpm0.v.col(0);
          for(int m = 0; m < N_moments.at(1); m+=MEMORY_alt)
            {
              
              // iterate MEMORY_alt times, just like before. No need to multiply by v here
              for(int i = m; i < m + MEMORY_alt; i++)
                kpm2.cheb_iteration(i);


	      //----------------------------ANY M-----------------------------//		  
	      int i_max = MEMORY_alt,
		j_max = MEMORY_alt;


	      if(n + MEMORY_alt > N_moments.at(0))
		i_max = N_moments.at(0) % MEMORY_alt;
	      if(m + MEMORY_alt > N_moments.at(1))
		j_max = N_moments.at(1) % MEMORY_alt;
	      //--------------------------------------------------------------//	  	      	      
              
              // Finally, do the matrix product and store the result in the Gamma matrix
              tmp.setZero();
	      
	      for(std::size_t ii = 0; ii < r.Sized ; ii += r.Ld[0])
	        tmp.block(0,0,i_max,j_max) += kpm3.v.block(ii,0, r.Ld[0], i_max).adjoint() * kpm2.v.block(ii, 0, r.Ld[0], j_max);

	      //Here, matrix multiplication result is updated on the global variable:	      
	     #pragma omp critical
	      {
	      for(int j = 0; j < j_max; j++)
                for(int i = 0; i < i_max; i++)               
	          Global.general_gamma_R.matrix()( n+i, m+j) += tmp(i,j);
	      }
	      
		  /*------------------------------------------------------------------*/
	      
      
            }
        }
      
    /* //--------------------------Parallelization of the general_gamma_R update. Not fully implemented/tested, but also not significantly impactful------// 
    //This is actually only valid for rows=cols, which is the case for dc_conductivity calls of Gamma2D;

    int id,  Nthrds, l_start, l_end, size, rows, cols;
    rows = Global.general_gamma.rows();
    cols = Global.general_gamma.cols();
    
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
        
    l_start = id * cols / Nthrds;
    l_end = (id+1) * cols / Nthrds;

    if (id == Nthrds-1  )  l_end = cols;

    size = l_end-l_start;


    Global.general_gamma.block(0,l_start,rows, size) += (Global.general_gamma_R.block(0,l_start,rows, size) - Global.general_gamma.block(0,l_start,rows, size))/value_type(average + 1);			
    Global.general_gamma_R.block(0,l_start,rows, size).setZero();
    average++;
     //----------------------------------------------------------------------------------------------------------------------------------------------------// */

      
      //Updating general_gamma_R with a single core instead:
#pragma omp barrier
      {
#pragma omp master
        {
          Global.general_gamma.matrix() += (Global.general_gamma_R.matrix() - Global.general_gamma.matrix())/value_type(average + 1);			
          Global.general_gamma_R.matrix().setZero();
          average++;
        }
      }

      
    } //end of randV cycle
  }//end of NDisorder cycles

  
#pragma omp barrier
  {
#pragma omp master
    {
      Global.general_gamma.matrix()  *= factor;
      Global.general_gamma_R.matrix() = Global.general_gamma.matrix(); //Here using Global.general_gamma_R as a temporary buffer; 
      Global.general_gamma.matrix()   = (Global.general_gamma_R.matrix()  +  factor * Global.general_gamma_R.matrix().adjoint())/2.0;
      store_gamma_2(name_dataset);
    }
  }
}


template <typename T,unsigned D>
void Simulation<T,D>::store_gamma_2( std::string name_dataset){
  debug_message("Entered store_gamma\n");
  // The whole purpose of this function is to take the Gamma matrix calculated by

  {
    H5::H5File * file = new H5::H5File(name, H5F_ACC_RDWR);
    write_hdf5(Global.general_gamma, file, name_dataset);
    delete file;
  }
    
  debug_message("Left store_gamma\n");
}



#define instantiate(type, dim)  template void Simulation<type,dim>::Gamma2D_mod(int, int, int, std::vector<int>, std::vector<std::vector<unsigned>>, std::string); \
  template void Simulation<type,dim>::store_gamma_2(std::string);
#include "instantiate.hpp"
