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
#include "queue.hpp"
#include "Simulation.hpp"
#include "Hamiltonian.hpp"
#include "KPM_VectorBasis.hpp"
#include "KPM_Vector.hpp"

template <typename T,unsigned D>
void Simulation<T,D>::calc_conddc_mod(){
    debug_message("Entered Simulation::calc_conddc_FFT\n");

    // Make sure that all the threads are ready before opening any files
    // Some threads could still be inside the Simulation constructor
    // This barrier is essential
#pragma omp barrier

    int NMoments, NRandom, NDisorder, direction, NReps;
  bool local_calculate_conddc_mod = false;
#pragma omp master
{
  H5::H5File * file = new H5::H5File(name, H5F_ACC_RDONLY);
  Global.calculate_conddc_mod = false;
  try{
    int dummy_variable;
    get_hdf5<int>(&dummy_variable,  file, (char *)   "/Calculation/conductivity_dc_mod/NumMoments");
    Global.calculate_conddc_mod = true;
  } catch(H5::Exception& e) {debug_message("CondDC_FFT: no need to calculate CondDC.\n");}
  file->close();
  delete file;
}
#pragma omp barrier
#pragma omp critical
  local_calculate_conddc_mod = Global.calculate_conddc_mod;

#pragma omp barrier
  
if(local_calculate_conddc_mod){
#pragma omp master
      {
        std::cout << "Calculating CondDC_mod.\n";
      }
#pragma omp barrier
#pragma omp critical
{
    H5::H5File * file = new H5::H5File(name, H5F_ACC_RDONLY);

    debug_message("DC conductivity: checking if we need to calculate DC_mod conductivity.\n");
    get_hdf5<int>(&direction, file, (char *) "/Calculation/conductivity_dc_mod/Direction");
    get_hdf5<int>(&NMoments, file, (char *)  "/Calculation/conductivity_dc_mod/NumMoments");
    get_hdf5<int>(&NRandom, file, (char *)   "/Calculation/conductivity_dc_mod/NumRandoms");
    get_hdf5<int>(&NDisorder, file, (char *) "/Calculation/conductivity_dc_mod/NumDisorder");
    get_hdf5<int>(&NReps, file, (char *) "/Calculation/conductivity_dc_mod/NumReps");
    
    file->close();
    delete file;

}
 CondDC_mod(NMoments, NRandom, NDisorder, NReps, direction);
 }

}
template <typename T,unsigned D>

void Simulation<T,D>::CondDC_mod(int NMoments, int NRandom, int NDisorder, int NReps, int direction){
  std::string dir(num2str2(direction));
  std::string dirc = dir.substr(0,1)+","+dir.substr(1,2);
  Gamma2D_mod(NRandom, NDisorder, NReps, {NMoments,NMoments}, process_string(dirc), "/Calculation/conductivity_dc_mod/Gamma"+dir);
}



template class Simulation<float ,1u>;
template class Simulation<double ,1u>;
template class Simulation<long double ,1u>;
template class Simulation<std::complex<float> ,1u>;
template class Simulation<std::complex<double> ,1u>;
template class Simulation<std::complex<long double> ,1u>;

template class Simulation<float ,3u>;
template class Simulation<double ,3u>;
template class Simulation<long double ,3u>;
template class Simulation<std::complex<float> ,3u>;
template class Simulation<std::complex<double> ,3u>;
template class Simulation<std::complex<long double> ,3u>;

template class Simulation<float ,2u>;
template class Simulation<double ,2u>;
template class Simulation<long double ,2u>;
template class Simulation<std::complex<float> ,2u>;
template class Simulation<std::complex<double> ,2u>;
template class Simulation<std::complex<long double> ,2u>;


