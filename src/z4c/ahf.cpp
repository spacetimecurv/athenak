//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ahf.cpp
//! \brief Implementation of the apparent horizon finder class
//!        based on the fast-flow algorithm of Gundlach:1997us and Alcubierre:1998rq

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <unistd.h>
#include <cmath> // NAN

#ifdef MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "ahf.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "utils/linear_algebra.hpp" // (OS): Missing, needed for metric inversion (in ADM?)
#include "utils/tensor.hpp" // (OS): Missing, not needed, since AthenaPointTensor exists in athena_tensor.hpp
#include "coordinates/coordinates.hpp"
#include "compact_object_tracker.hpp" // (OS): instead of puncture_tracker.hpp and extrema_tracker.hpp

//----------------------------------------------------------------------------------------
//! \fn AHF::AHF(Mesh * pmesh, ParameterInput * pin, int n)
//! \brief Class for apparent horizon finder
AHF::AHF(Mesh *pmesh, ParameterInput *pin, int n):
  pmesh(pmesh),
  pin(pin),
  th_grid("th_grid",1), ph_grid("ph_grid",1), weights("weights",1,1)// (OS): Initialize Device Array
  P("P",1,1), dPdth("dPdth",1,1), dPdth2("dPdth2",1,1), // (OS): Initialize Device Array  
  Y0("Y0",1,1,1), Ys("Ys",1,1,1), Yc("Yc",1,1,1), // (OS): Initialize Device Array  
  dY0dth("dY0dth",1,1,1), dYcdth("dYcdth",1,1,1), dYsdth("dYsdth",1,1,1), 
  dYcdph("dYcdph",1,1,1), dYsdph("dYsdph",1,1,1), dY0dth2("dY0dth2",1,1,1), 
  dYcdth2("dYcdth2",1,1,1), dYcdthdph("dYcdthdph",1,1,1), dYsdth2("dYsdth2",1,1,1), 
  dYsdthdph("dYsdthdph",1,1,1), dYcdph2("dYcdph2",1,1,1), dYsdph2("dYsdph2",1,1,1),
  a0("a0",1), ac("ac",1), as("as",1), // (OS): Initialize Device Array
  rr("rr",1,1), rr_dth("rr_dth",1,1), rr_dph("rr_dph",1,1), // (OS): Initialize Device Array
  rho("rho",1,1) // (OS): Initialize Device Array
{
  nh = n; // The n-th horizon
  std::string parname;
  std::string n_str = std::to_string(nh);

  // Read parameter input variables
  nhorizon = pin->GetOrAddInteger("ahf", "num_horizons", 1); // Number of horizons
  ntheta = pin->GetOrAddInteger("ahf", "ntheta", 5); // Number of points theta
  nphi = pin->GetOrAddInteger("ahf", "nphi", 10); // Number of points phi
  if ((nphi + 1) % 2 == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "nphi must be even, but is " << nphi << std::endl;
    exit(EXIT_FAILURE);
  }

  lmax = pin->GetOrAddInteger("ahf", "lmax", 4);
  lmax1 = lmax + 1;

  parname = "flow_iterations_";
  parname += n_str;
  flow_iterations = pin->GetOrAddInteger("ahf", parname, 100);

  parname = "flow_alpha_beta_const_"; 
  parname += n_str;
  flow_alpha_beta_const = pin->GetOrAddReal("ahf", parname, 1.0);

  parname = "hmean_tol_";
  parname += n_str;
  hmean_tol = pin->GetOrAddReal("ahf", parname, 100.);

  parname = "mass_tol_";
  parname += n_str;
  mass_tol = pin->GetOrAddReal("ahf", parname, 1e-2);

  verbose = pin->GetOrAddBoolean("ahf", "verbose", false);
  root = pin->GetOrAddInteger("ahf", "mpi_root", 0);
  merger_distance = pin->GetOrAddReal("ahf", "merger_distance", 0.1);
  use_stored_metric_drvts = pin->GetBoolean("z4c", "store_metric_drvts");

  // Initial guess
  parname = "initial_radius_";
  parname += n_str;
  initial_radius = pin->GetOrAddReal("ahf", parname, 1.0);
  rr_min = -1.0;

  expand_guess = pin->GetOrAddReal("ahf", "expand_guess", 1.0);
  npunct = pin->GetOrAddInteger("z4c", "npunct", 0);

  // Center
  parname = "center_x";
  parname += n_str;
  center[0] = pin->GetOrAddReal("ahf", parname, 0.0);
  parname = "center_y";
  parname += n_str;
  center[1] = pin->GetOrAddReal("ahf", parname, 0.0);
  parname = "center_z";
  parname += n_str;
  center[2] = pin->GetOrAddReal("ahf", parname, 0.0);

  parname = "use_puncture_";
  parname += n_str;
  use_puncture = pin->GetOrAddInteger("ahf", parname, -1);

  compute_every_iter = 1; // (OS): Is this covered by the task triggers?

  if (use_puncture >= 0) {
    // Center is determined on the fly during the initial guess
    // to follow the chosen puncture
    if (use_puncture >= npunct) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " : punc = " << use_puncture << " > npunct = " << npunct << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  parname = "use_puncture_massweighted_center_";
  parname += n_str;
  use_puncture_massweighted_center = pin->GetOrAddInteger("ahf", parname, -1);

  parname = "use_extrema_";
  parname += n_str;
  use_extrema = pin->GetOrAddInteger("ahf", parname, -1);

  if (use_extrema >= 0) {
    const int N_tracker = pmesh->ptracker_extrema->N_tracker; // (OS): Change this!
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " : extrema = " << use_extrema << " > N_tracker = " << N_tracker << std::endl;
    exit(EXIT_FAILURE);
  }

  parname = "start_time_";
  parname += n_str;
  start_time = pin->GetOrAddReal("ahf", parname, std::numeric_limits<double>::max());

  parname = "stop_time_";
  parname += n_str;
  stop_time = pin->GetOrAddReal("ahf", parname, -1.0);

  parname = "wait_until_punc_are_close_";
  parname += n_str;
  wait_until_punc_are_close = pin->GetOrAddBoolean("ahf", parname, 0);

  // Grid and quadrature weights
  std::string quadrature = pin->GetOrAddString("ahf","quadrature","gausslegendre");
  SetGridWeights(quadrature); 

  // Initialize last & found
  parname = "last_a0_";
  parname += n_str;
  last_a0 = pin->GetOrAddReal("ahf", parname, -1);

  parname = "ah_found_a0_";
  parname += n_str;
  ah_found = pin->GetOrAddBoolean("ahf", parname, false);

  parname = "time_first_found_";
  parname += n_str;
  time_first_found = pin->GetOrAddReal("ahf", parname, -1.0);

  // Points for spherical harmonics l >= 1
  lmpoints = lmax1 * lmax1;

  // Reallocate for the coefficients
  Kokkos::realloc(a0, lmax1); // (OS): Is this right? Should be!
  Kokkos::realloc(ac, lmpoints);
  Kokkos::realloc(as, lmpoints);

  // Reallocate for the spherical harmonics
  // The spherical grid is the same for all surfaces
  Kokkos::realloc(Y0, ntheta, nphi, lmax1);
  Kokkos::realloc(Yc, ntheta, nphi, lmpoints);
  Kokkos::realloc(Ys, ntheta, nphi, lmpoints);

  Kokkos::realloc(dY0dth, ntheta, nphi, lmax1);
  Kokkos::realloc(dYcdth, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYsdth, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYcdph, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYsdph, ntheta, nphi, lmpoints);
  
  Kokkos::realloc(dY0dth2, ntheta, nphi, lmax1);
  Kokkos::realloc(dYcdth2, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYcdthdph, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYsdth2, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYsdthdph, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYcdph2, ntheta, nphi, lmpoints);
  Kokkos::realloc(dYsdph2, ntheta, nphi, lmpoints);
  
  // NEXT: spherical harmonics
} 

//----------------------------------------------------------------------------------------
//! \fn void AHF::SetGridWeights()
//! \brief Set nodes on the sphere & weights for the 2D integrals 
void AHF::SetGridWeights(std::string method)
{
  if ((nphi + 1) % 2 == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "nphi must be even, but is " << nphi << std::endl;
    exit(EXIT_FAILURE);
  }

  Kokkos::realloc(th_grid, ntheta);
  Kokkos::realloc(ph_grid, nphi);
  Kokkos::realloc(weights, ntheta, nphi);

  if (method == "sums") {
    const Real dphi = 2.0 * M_PI / nphi;
    for (int j = 0; j < nphi; ++j) {
      ph_grid(j) = dphi * (0.5 + j);
    }

    const Real dtheta = M_PI / ntheta;
    for (int i = 0; i < ntheta; ++i) {
      th_grid(i) = dtheta * (0.5 + i);
    }

    for (int i = 0; i < ntheta; ++i) {
      const Real dcosth = std::sin(th_grid(i)) * dtheta;

      for (int j = 0; j < nphi; ++j) {
        weights(i,j) = dcosth * dphi;
      }
    }
  } 
  else if (method == "gausslegendre") {
    if (ntheta != nphi / 2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "ntheta = " << ntheta << " should be nphi/2 = " << nphi/2 << std::endl;
      exit(EXIT_FAILURE);
    }

    const Real dphi = 2.0 * M_PI / nphi;
    for (int j = 0; j < nphi; ++j) {
      ph_grid(j) = dphi * (0.5 + j);
    }
    
    Real *gl_weights = new Real[ntheta];
    Real *gl_nodes = new Real[ntheta];

    GLQuad_Nodes_Weights(-1.0, 1.0, gl_nodes, gl_weights, ntheta);

    for (int i = 0; i < ntheta; ++i) {
      th_grid(i) = std::acos(gl_nodes[i]);
    }

    for (int i = 0; i < ntheta; ++i) {
      for (int j = 0; j < nphi; ++j) {
        weights(i,j) = gl_weights[i] * dphi;
      }
    }

    delete[] gl_weights;
    delete[] gl_nodes;
  }
  else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Unknown method " << method << std::endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn int AHF::GLQuad_Nodes_Weights(const Real a, const Real b, Real * x, Real * w, const int n)
//! \brief Nodes and weights for Gauss-Legendre quadrature
void AHF::GLQuad_Nodes_Weights(const Real a, const Real b, Real *x, Real *w, const int n)
{
  #define SMALL (1e-14)
  Real z1, z, xm, xl, pp, p3, p2, p1;
  const int m = (n + 1) / 2;
  xm = 0.5 * (b + a);
  xl = 0.5 * (b - a);

  for (int i = 1; i <= m; i++) {
    z = std::cos(M_PI * (i - 0.25) / (n + 0.5));

    do {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
    } while (std::fabs(z - z1) > SMALL);
    x[i-1] = xm - xl * z;
    x[n-i] = xm + xl * z;
    w[i-1] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
    w[n-i] = w[i-1];
  }
}