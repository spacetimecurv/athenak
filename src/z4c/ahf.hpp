//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ahf.hpp
//! \brief Basic functionality for the AHF class.

#ifndef AHF_HPP
#define AHF_HPP

#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "z4c.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

//! \class AHF
//! \brief Apparent Horizon Finder class
class AHF {
public:
  // Constructor for AHF object
  AHF(Mesh *pmesh, ParameterInput *pin, int n);

  // Default Destructor for AHF object (closes output file)
  ~AHF();

  void Find(int iter, Real time); // main functionality for finding AH
  void Write(int iter, Real time); // function for result writing

  bool CalculateMetricDerivatives(int iter, Real time);
  bool DeleteMetricDerivatives(int iter, Real time);

  Real GetHorizonRadius() const { return ah_prop[hmeanradius]; }

  // Some of the main parameters in the fast-flow algorithm
  bool ah_found; // Horizon found
  Real time_first_found; // Time, when horizon first found
  Real initial_radius; // Initial guess for the radius of the horizon
  Real rr_min; // Minimum radius
  Real expand_guess; // Expand the initial guess by this factor
  Real center[3]; // Center around which the horizon is searched

  // Fast-Flow parameters
  Real hmean_tol; // for convergence 
  Real mass_tol; // for convergence
  int flow_iterations; // number of flow iterations
  Real flow_alpha_beta_const; // alpha & beta constants in the iteration formula
                              // Eqs. (43) & (44) of https://arxiv.org/pdf/gr-qc/9707050
  bool verbose;

  // Spherical harmonics & Legendre polynomials
  int lmax; // Multipoles
  int ntheta, nphi; // Grid points

  // Compact Object Tracker variables
  int use_puncture; // n surface follows the puncture tracker if use_puncture[n] > 0
  bool use_puncture_massweighted_center; // n surface uses the punctures' mass-weighted center
  int use_extrema; // n surface follows the extrema tracker if use_extrema[n] > 0
  Real merger_distance; // Distance in M at which BHs are considered as merged

  // Start and Stop times for each surface
  Real start_time;
  Real stop_time;

  // Compute every n iterations
  int compute_every_iter;

private:
  int npunct; // Number of punctures
  int lmax1; // lmax + 1
  int lmpoints; // lmax * lmax
  int nh; // Counter variable
  bool wait_until_punc_are_close;
  bool use_stored_metric_drvts;
  int nstart, nhorizon; // Number of horizons
  int fastflow_iter = 0;

  // Metric interpolation order
  metric_interp_order = 2 * NGHOST - 1 // (OS): Use pmesh to access indcs.ng

  // Arrays for the grid and quadrature weights
  DvceArray1D<Real> th_grid, ph_grid; // (OS): Device or Host? Host
  DvceArray2D<Real> weights; // (OS): Device or Host?

  // Arrays of Legendre polynomials and derivatives
  DvceArray2D<Real> P, dPdth, dPdth2; // (OS): Device or Host?

  // Arrays of spherical harmonics and derivatives
  DvceArray3D<Real> Y0, Yc, Ys; // (OS): Device or Host? Maybe Device if expansion loop parallelized
  DvceArray3D<Real> dY0dth, dYcdth, dYsdth, dYcdph, dYsdph; // (OS): Device or Host?
  DvceArray3D<Real> dY0dth2, dYcdth2, dYcdthdph, dYsdth2, dYsdthdph, dYcdph2, dYsdph2; // (OS): Device or Host?

  // Arrays for spectral coefficients
  DvceArray1D<Real> a0; // (OS): Device or Host?
  DvceArray1D<Real> ac; // (OS): Device or Host?
  DvceArray1D<Real> as; // (OS): Device or Host?
  Real last_a0; // last coefficient a_00

  // Arrays used for the fields on the sphere
  AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> g; // (OS): AthenaPointTensor or AthenaTensor?
  AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> K; // (OS): AthenaPointTensor or AthenaTensor?
  AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 3> dg; // (OS): AthenaPointTensor or AthenaTensor?
  DvceArray2D<Real> rr, rr_dth, rr_dph; // (OS): Device or Host?
  // (OS): Fix NDIM & NGHOST

  // Array computed in Surface Integrals
  DvceArray2D<Real> rho;

  // Indexes of surface integrals
  enum {
    iarea,
    icoarea,
    ihrms,
    ihmean,
    iSx, iSy, iSz,
    invar
  };
  Real integrals[invar]; // Array of surface integrals

  // Indexes of horizon quantities
  enum{
    harea,
    hcoarea,
    hhrms,
    hhmean,
    hSx, hSy, hSz, hS,
    hmass,
    hmeanradius,
    hminradius,
    hnvar
  };
  Real ah_prop[hnvar]; // Array of horizon quantities

  // Flag points
  DvceArray2D<int> havepoint; // (OS): Device or Host?

  // Functions used in the fast-flow algorithm
  void MetricDerivatives(MeshBlock *pmb); // (OS): MeshBlockPack?
  void MetricInterp(MeshBlock *pmb); // (OS): MeshBlockPack?
  void SurfaceIntegrals();
  void FastFlowLoop();
  void UpdateFlowSpectralComponents();
  void RadiiFromSphericalHarmonics();
  void InitialGuess();
  void ComputeSphericalHarmonics();
  void ComputeLegendre(const Real theta); 
  int lmindex(const int l, const int m);
  int tpindex(const int i, const int j);
  void GLQuad_Nodes_Weights(const Real a, const Real b, Real *x, Real *w, const int n);
  void SetGridWeights(std::string method);
  void factorial_list(Real *fac, const int maxn);

  // Pointers to Mesh and ParameterInput
  Mesh const *pmesh;
  ParameterInput *pin;

  // Control parameters
  int root;
  int ioproc;
  std::string ofname_summary;
  std::string ofname_shape;
  std::string ofname_verbose;
  FILE *pofile_summary;
  FILE *pofile_shape;
  FILE *pofile_verbose;

  // Functions to interface with puncture tracker
  Real PuncMaxDistance(); 
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc);
  bool PuncAreClose();
};

#endif