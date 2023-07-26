/***********************************************************/
/*                                                         */
/*   Copyright (C) 2018-2022, M. Andelkovic, L. Covaci,    */
/*  A. Ferreira, S. M. Joao, J. V. Lopes, T. G. Rappoport  */
/*                                                         */
/***********************************************************/

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <string>
#include <omp.h>

#include "H5Cpp.h"
#include "../Tools/ComplexTraits.hpp"
#include "../Tools/myHDF5.hpp"

#include "../Tools/parse_input.hpp"
#include "../Tools/systemInfo.hpp"
#include "conductivity_dc_FFT.hpp"
#include "../Tools/functions.hpp"

#include "../macros.hpp"

template <typename T, unsigned DIM>
conductivity_dc_FFT<T, DIM>::conductivity_dc_FFT(system_info<T, DIM>& info, shell_input & vari){
    /* Class constructor
     * Finds all the information needed to compute the DC conductivity
     * but does not compute it. To calculate the DC conductivity, the method 
     * calculate() needs to be called. If there is not enough data to compute
     * the DC conductivity, the code will exit with an error.
     */

    H5::Exception::dontPrint();
    units = unit_scale;     // units of the DC conductivity
    systemInfo = info;      // retrieve the information about the Hamiltonian
    variables = vari;       // retrieve the shell input

    isPossible = false;         // do we have all we need to calculate the conductivity?
    isRequired = is_required() && variables.CondDC_is_required; // was this quantity (conductivity_dc_FFT) asked for?


    // If the DC conductivity was requested, all the necessary parameters for its
    // computation will be set and printed out to std::cout
    if(isRequired){
        set_default_parameters();        // sets a default set of paramters for the calculation
        isPossible = fetch_parameters(); // finds all the paramters in the .h5 file
        override_parameters();           // overrides parameters with the ones from the shell input
        set_energy_limits();             // sets the energy limits used in the integration


        if(isPossible){
          printDC();                  // Print all the parameters used
          calculate2();
        } else {
          std::cout << "ERROR. The DC conductivity was requested but the data "
              "needed for its computation was not found in the input .h5 file. "
              "Make sure KITEx has processed the file first. Exiting.";
          exit(1);
        }
    }
}

template <typename T, unsigned DIM>
bool conductivity_dc_FFT<T, DIM>::is_required(){
    // Checks whether the DC conductivity has been requested
    // by analysing the .h5 config file. If it has been requested, 
    // some fields have to exist, such as "Direction"


    // Make sure the config filename has been initialized
    std::string name = systemInfo.filename.c_str();
    if(name == ""){
        std::cout << "ERROR: Filename uninitialized. Exiting.\n";
        exit(1);
    }

    // location of the information about the conductivity
    char dirName[] = "/Calculation/conductivity_dc_FFT/Direction";
	H5::H5File file = H5::H5File(name, H5F_ACC_RDONLY);

    // Check if this dataset exists
    bool result = false;
    try{
        get_hdf5(&direction, &file, dirName);
        result = true;
    } catch(H5::Exception& e){}


    file.close();
    return result;
}
	
template <typename T, unsigned DIM>
void conductivity_dc_FFT<T, DIM>::set_default_parameters(){
    // Sets default values for the parameters used in the 
    // calculation of the DC conductivity. These are the parameters
    // that will be overwritten by the config file and the
    // shell input parameters. 

    double scale = systemInfo.energy_scale;
    double shift = systemInfo.energy_shift;



    NFermiEnergies      = 100;
    minFermiEnergy      = -1.0;//(-1.0 - shift)/scale;        // -1eV in KPM reduced units [-1,1]
    maxFermiEnergy      = 1.0;//( 1.0 - shift)/scale;        //  1eV in KPM reduced units [-1,1]
    default_NFermi      = true;
    default_mFermi      = true;
    default_MFermi      = true;

    default_NumMoments  = true;

    // integrate means to use the full energy range in the integration
    full_range          = 1;
    default_full_range  = true;
    NumThreads          = systemInfo.NumThreads;
    default_NumThreads  = true;

    NEnergies           = 100;       // Number of energies used in the energy integration
    default_NEnergies   = true;

    deltascat           = 0.00/scale;       // scattering parameter in the delta function
    scat                = 0.00/scale;       // scattering parameter of 10meV in
    default_scat        = true;             // the Green's functions in KPM reduced units
    default_deltascat   = true;             // the Green's functions in KPM reduced units

    filename            = "condDC_FFT.dat";     // Filename to save the final result
    default_filename    = true;

    // Temperature is in energy units, so it is actually kb*T, where kb is Boltzmann's constant
    temperature         = 0.000/scale;      
    beta                = 1.0/temperature;
    default_temp        = true;
}

template <typename T, unsigned DIM>
void conductivity_dc_FFT<T, DIM>::set_energy_limits(){
    // Attempts to find the energy limits from the information
    // about the density of states stored in the systemInfo class.
    // If it cant, uses default limits.


    minEnergy               = -0.99;    // Minimum energy
    maxEnergy               = 0.99;     // Maximum energy
    default_energy_limits   = true;

    // Choose whether or not to use the limits of integration as
    // computed from the density of states. 
    if(systemInfo.EnergyLimitsKnown and !full_range){
        minEnergy = systemInfo.minEnergy;
        maxEnergy = systemInfo.maxEnergy;
        default_energy_limits = false;
    }
}

template <typename T, unsigned DIM>
bool conductivity_dc_FFT<T, DIM>::fetch_parameters(){
	debug_message("Entered conductivit_dc::read.\n");
	//This function reads all the data from the hdf5 file that's needed to 
    //calculate the dc conductivity
	 
	H5::H5File file;
  std::string dirName = "/Calculation/conductivity_dc_FFT/";  // location of the information about the conductivity
	file = H5::H5File(systemInfo.filename.c_str(), H5F_ACC_RDONLY);

  // Fetch the direction of the conductivity and convert it to a string
  get_hdf5(&direction, &file, (char*)(dirName+"Direction").c_str());
  std::string dirString = num2str2(direction);

  // Fetch the temperature from the .h5 file
  // The temperature (kb*T) is in energy units. It is already reduced by SCALE from within the python script
	get_hdf5(&temperature, &file, (char*)(dirName+"Temperature").c_str());	
  beta = 1.0/temperature;
  default_temp = false;


  // Fetch the number of Chebyshev Moments
  int NRandom, NDisorder;

  get_hdf5(&MaxMoments, &file, (char*)(dirName+"NumMoments" ).c_str());	
  get_hdf5(&NRandom,    &file, (char*)(dirName+"NumRandoms" ).c_str());
  get_hdf5(&NDisorder,  &file, (char*)(dirName+"NumDisorder").c_str());

  RD=NRandom*NDisorder;
	/*Shouldt be used
  // Fetch the number of Fermi energies from the .h5 file
	get_hdf5(&NFermiEnergies, &file, (char*)(dirName+"NumPoints").c_str());	
  default_NFermi = false;
	*/
  NumMoments     = MaxMoments;
  NFermiEnergies = NumMoments;
  NEnergies      = NumMoments;       // Number of energies used in the energy integration

  default_NumMoments = true;

  default_NFermi    = false; //Updated NFermi and NEnergies default
  default_NEnergies = false;

  // Check whether the matrices we're going to retrieve are complex or not
  int complex = systemInfo.isComplex;


  
  // Retrieve the integrand Matrix
  std::string MatrixName = dirName + "integrand" + dirString;
  bool possible = false;
  try{
    debug_message("Filling the integrand matrix.\n");
    integrand_FFT = Eigen::Array<double,-1,-1>::Zero(NumMoments,1);

    get_hdf5(integrand_FFT.data(), &file, (char*)MatrixName.c_str());

    possible = true;
  } catch(H5::Exception& e) {
      debug_message("Conductivity DC: There is no Gamma matrix.\n");
  }
  
  
    // Retrieve the R_vecs Matrix
  integrand_FFT_R_data = Eigen::Array<double,-1,-1>::Zero(NumMoments,RD); 
  
  for(int rd=1; rd<=RD; rd++){
    std::string  RDMatrixName = dirName + "integrand" + dirString+"_RD"+std::to_string(rd);
    bool possible = false;
    try{
      debug_message("Filling the integrand RD matrix.\n");
      Eigen::Array<double,-1,-1> integrand_R_data_col = Eigen::Array<double,-1,-1>::Zero(NumMoments,1);

      get_hdf5(integrand_R_data_col.data(), &file, (char*)RDMatrixName.c_str());
    
      integrand_FFT_R_data.matrix().col(rd-1)=integrand_R_data_col.matrix();

      possible = true;
    } catch(H5::Exception& e) {
        debug_message("Conductivity DC: There is no Integrand_RD matrix.\n");
    }
  }


  

  file.close();
	debug_message("Left conductivity_dc::read.\n");
  return possible;
}

template <typename U, unsigned DIM>
void conductivity_dc_FFT<U, DIM>::override_parameters(){
    // Overrides the current parameters with the ones from the shell input.
    // These parameters are in eV or Kelvin, so they must scaled down
    // to the KPM units. This includes the temperature

    double scale = systemInfo.energy_scale;
    double shift = systemInfo.energy_shift;

    if(variables.CondDC_Temp != -1){
        temperature     = variables.CondDC_Temp/scale;
        beta            = 1.0/temperature;
        default_temp    = false;
    }

    if(variables.CondDC_NumEnergies != -1){
        NEnergies = variables.CondDC_NumEnergies;
        default_NEnergies = false;
    }

    if(variables.CondDC_nthreads != -1){
        NumThreads = variables.CondDC_nthreads;
        default_NumThreads = false;

        if(NumThreads < 1){
          std::cout << "NumThreads cannot be smaller than 1. Aborting.\n";
          assert(NumThreads > 0);
          exit(1);
        }
    }

    if(variables.CondDC_NumMoments != -1){
        NumMoments = variables.CondDC_NumMoments;
        default_NumMoments = false;

        if(NumMoments > MaxMoments){
          std::cout << "NumMoments cannot be larger than the number of Chebyshev ";
          std::cout << "moments computed with KITEx. Aborting.\n";
            
          assert(NumMoments <= MaxMoments);
          exit(1);
        }
    }

    // integrate = true means to use the full energy range in the integration
    if(variables.CondDC_integrate != -1){
        full_range = variables.CondDC_integrate;
        default_full_range = false;
    }


    if(variables.CondDC_Scat != -8888){
        scat            = variables.CondDC_Scat/scale;
        default_scat    = false;
    }

    if(variables.CondDC_deltaScat != -8888){
        deltascat         = variables.CondDC_deltaScat/scale;
        default_deltascat = false;
    } else {
        deltascat = scat;
    }


    if(variables.CondDC_FermiMin != -8888){
        minFermiEnergy  = (variables.CondDC_FermiMin - shift)/scale;
        default_mFermi   = false;
    }

    if(variables.CondDC_FermiMax != -8888){  
        maxFermiEnergy  = (variables.CondDC_FermiMax - shift)/scale;
        default_MFermi   = false;
    }

    if(variables.CondDC_NumFermi != -1){
        NFermiEnergies  = variables.CondDC_NumFermi;
        default_NFermi   = false;
    }

    if(variables.CondDC_Name != ""){
        filename            = variables.CondDC_Name;
        default_filename    = false;
    }
    

}


template <typename U, unsigned DIM>
void conductivity_dc_FFT<U, DIM>::printDC(){
    double scale = systemInfo.energy_scale;
    double shift = systemInfo.energy_shift;
    std::string energy_range = "[" + std::to_string(minEnergy*scale + shift) + ", " + std::to_string(maxEnergy*scale + shift) + "]";

    // Prints all the information about the parameters
    std::cout << "The DC conductivity (FFT algo.) will be calculated with these parameters: (eV, Kelvin)\n"
        "   Temperature: "             << temperature*scale             << ((default_temp)?         " (default)":"") << "\n"
        "   Broadening: "              << scat*scale                    << ((default_scat)?         " (deactivated)":"") << "\n"
        "   Delta broadening: "        << deltascat*scale               << ((default_deltascat)?    " (deactivated)":"") << "\n"
        "   Max Fermi energy: "        << maxFermiEnergy*scale + shift  << ((default_MFermi)?       " (default)":"") << "\n"
        "   Min Fermi energy: "        << minFermiEnergy*scale + shift  << ((default_mFermi)?       " (default)":"") << "\n"
        "   Number Fermi energies: "   << NFermiEnergies                << ((default_NFermi)?       " (default)":"") << "\n"
        "   Filename: "                << filename                      << ((default_filename)?     " (default)":"") << "\n"
        "   Integration range: "       << energy_range                  << ((default_energy_limits)?" (default)":" (Estimated from DoS)") << "\n"
        "   Num integration points: "  << NEnergies                     << ((default_NEnergies)?    " (default)":"") << "\n"
        "   Num Chebychev moments: "   << NumMoments                    << ((default_NumMoments)?   " (default)":"") << "\n"
        "   Num threads: "             << NumThreads                    << ((default_NumThreads)?   " (default)":"") << "\n"; 
}



template <typename U, unsigned DIM>
void conductivity_dc_FFT<U, DIM>::calculate2(){
  fermiEnergies = Eigen::Matrix<U, -1, 1>::Zero(NFermiEnergies,1);//LinSpaced(NFermiEnergies, minFermiEnergy, maxFermiEnergy);
  energies = Eigen::Matrix<U, -1, 1>::Zero(NEnergies,1);//LinSpaced(NFermiEnergies, minFermiEnergy, maxFermiEnergy);

  
  for(int i=0; i<NumMoments/2; i++) {
     U tmp = integrand_FFT(i);
     integrand_FFT(i) = integrand_FFT(NumMoments-i-1);
     integrand_FFT(NumMoments-i-1) = tmp;
  }
  
  for(int i=0; i<NumMoments; i++) {     
     energies(i) = cos(M_PI*(double(NumMoments-i-1)+0.5)/NumMoments);
     fermiEnergies(i) = energies(i);     
  }
  

  // integrate over the whole energy range for each Fermi energy
  Eigen::Matrix<std::complex<U>, -1, 1> condDC;
  U den  = -systemInfo.num_orbitals*systemInfo.spin_degeneracy/systemInfo.unit_cell_area/units; 
  condDC = calc_cond(integrand_FFT)*den;

  // save to a file
  save_to_file(condDC);
};



template <typename U, unsigned DIM>
Eigen::Matrix<std::complex<U>, -1, 1> conductivity_dc_FFT<U, DIM>::calc_cond(Eigen::Matrix<double, -1, -1> integrand){

  Eigen::Matrix<std::complex<U>, -1, 1> condDC;
  Eigen::Matrix<std::complex<U>, -1, 1> local_integrand;


  condDC = Eigen::Matrix<std::complex<U>, -1, 1>::Zero(NFermiEnergies, 1);
  local_integrand = Eigen::Matrix<std::complex<U>, -1, 1>::Zero(NEnergies, 1);


  U fermi;
  for(int i = 0; i < NFermiEnergies; i++){
    fermi = fermiEnergies(i);
    for(int j = 0; j < NEnergies; j++){
      local_integrand(j) = integrand(j)*fermi_function(energies(j), fermi, beta);
    }
    condDC(i) = integrate(energies, local_integrand);
  }

  return condDC;
}



template <typename U, unsigned DIM>
void conductivity_dc_FFT<U, DIM>::save_to_file(Eigen::Matrix<std::complex<U>, -1, -1> condDC){

  std::complex<U> cond;
  U energy;
  std::ofstream myfile;
  myfile.open(filename);

  for(int i=0; i < NFermiEnergies; i++){
    energy = fermiEnergies(i)*systemInfo.energy_scale + systemInfo.energy_shift;
    cond = condDC(i);
    myfile  << energy << " " << cond.real() << " " << cond.imag() << "\n";
  }
  
  myfile.close();

  
  std::ofstream myfile2;
  myfile2.open("condDC_integrand_R_data.dat");
  myfile2<<integrand_FFT_R_data.matrix();  
  myfile2.close();

}



// Instantiations
template class conductivity_dc_FFT<float, 1u>;
template class conductivity_dc_FFT<float, 2u>;
template class conductivity_dc_FFT<float, 3u>;

template class conductivity_dc_FFT<double, 1u>;
template class conductivity_dc_FFT<double, 2u>;
template class conductivity_dc_FFT<double, 3u>;

template class conductivity_dc_FFT<long double, 1u>;
template class conductivity_dc_FFT<long double, 2u>;
template class conductivity_dc_FFT<long double, 3u>;
