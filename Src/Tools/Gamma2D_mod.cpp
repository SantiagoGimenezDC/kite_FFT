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
void Simulation<T,D>::Gamma2D_mod(int NRandomV, int NDisorder, int NReps, std::vector<int> N_moments, 
                              std::vector<std::vector<unsigned>> indices, std::string name_dataset){
  int MEMORY_alt = N_moments.at(0)/NReps;    
  Eigen::Matrix<T, -1, -1> tmp(MEMORY_alt,MEMORY_alt); 
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

  Eigen::Array<T, -1, -1> gamma = Eigen::Array<T, -1, -1 >::Zero(1, size_gamma);

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
      // run through the left loop MEMORY iterations at a time
      for(int n = 0; n < N_moments.at(0); n+=MEMORY_alt)
        {
          
          // Iterate MEMORY times. The first time this occurs, we must exclude the zeroth
          // case, because it is already calculated, it's the identity
          for(int i = n; (i < n + MEMORY_alt && i< N_moments.at(0))  ; i++) {
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
	      
              // iterate MEMORY times, just like before. No need to multiply by v here
              for(int i = m; (i < m + MEMORY_alt && i<N_moments.at(1)  ); i++)
	          kpm2.cheb_iteration(i);

		  
		  /*----------------------------ANY M FIX-----------------------------*/
		  
	      int i_max = MEMORY_alt,
		j_max = MEMORY_alt;


	      if(n + MEMORY_alt > N_moments.at(0))
		i_max = N_moments.at(0) % MEMORY_alt;
	      if(m + MEMORY_alt > N_moments.at(1))
		j_max = N_moments.at(1) % MEMORY_alt;


	      // Finally, do the matrix product and store the result in the Gamma matrix
              tmp.setZero();


	      for(std::size_t ii = 0; ii < r.Sized ; ii += r.Ld[0])
	        tmp.block(0,0,i_max,j_max) += kpm3.v.block(ii,0, r.Ld[0], i_max).adjoint() * kpm2.v.block(ii, 0, r.Ld[0], j_max);

	      
		T flatten;
                long int ind;
                for(int j = 0; j < j_max; j++)
                  for(int i = 0; i < i_max; i++){		    
                      flatten = tmp(i,j);
                      ind = (m+j)*N_moments.at(0) + n+i;
                      gamma(ind) += (flatten - gamma(ind))/value_type(average + 1);		    
                  }
		    
		  /*------------------------------------------------------------------*/
		    


		    
            }
        }
      average++;
    }
  } 
  gamma = gamma*factor;

  store_gamma(&gamma, N_moments, indices, name_dataset);
}


#define instantiate(type, dim)  template void Simulation<type,dim>::Gamma2D_mod(int, int, int, std::vector<int>, std::vector<std::vector<unsigned>>, std::string); 
#include "instantiate.hpp"

