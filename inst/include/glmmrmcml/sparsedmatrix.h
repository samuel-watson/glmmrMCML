#ifndef SPARSEDMATRIX_H
#define SPARSEDMATRIX_H

#include <SparseChol.h>
#include <glmmr.h>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
class SparseDMatrix {
public:
  SparseDMatrix(glmmr::DData* data,
                const Eigen::ArrayXd& gamma,
                const Eigen::ArrayXi &Ap,
                const Eigen::ArrayXi &Ai):
   data_(data), gamma_(gamma){
    Ap_.resize(Ap.size());
    Eigen::Map<Eigen::ArrayXi>(Ap_.data(),Ap_.size()) = Ap;
    Ai_.resize(Ai.size());
    Eigen::Map<Eigen::ArrayXi>(Ai_.data(),Ai_.size()) = Ai;
    nx_ = Ai.size();
    n_ = data->N();
    Ax_ = std::vector<double>(nx_);
    mat_ = new sparse(Ap_);
    mat_->Ai = Ai_;
    chol_ = new SparseChol(mat_);
    update_parameters(gamma_);
  }
  
  glmmr::DData* data_;
  Eigen::ArrayXd gamma_;
  std::vector<int> Ap_;
  std::vector<int> Ai_;
  int nx_;
  int n_;
  std::vector<double> Ax_;
  sparse* mat_;
  SparseChol* chol_;
  
  void update_parameters(const Eigen::ArrayXd& gamma){
    gamma_ = gamma; // assign new parameter values
    int llim = 0;
    int nj = 0;//(int)N_dim_(0);
    int ulim = Ap_[nj+data_->cov_(0,1)];
    int j = 0;
    data_->subdata(0);
    
    for(int b=0; b < data_->B_; b++){
      DSubMatrix* dblock;
      dblock = new DSubMatrix(b, data_, gamma_);
      
      for(int i = llim; i<ulim; i++){
        if(i == Ap_[j+1])j++;
        
        Ax_[i] = dblock->get_val(Ai_[i]-nj,j-nj);
      }
      llim = ulim;
      if(b<(data_->B_-1)){
        nj += data_->n_dim();
        data_->subdata(b+1);
        ulim = Ap_[nj+data_->n_dim()];
      } 
      if(b == (data_->B_-1)){
        ulim = nx_;
      }
      delete dblock;
    }
    mat_->Ax = Ax_;
    chol_->ldl_numeric();
  }
  
  void update_parameters(const Eigen::VectorXd& gamma){
    Eigen::ArrayXd gammaa = gamma.array();
    update_parameters(gammaa);
  }
  
  void update_parameters(const std::vector<double>& gamma){
    std::vector<double> par = gamma;
    Eigen::ArrayXd gammaa = Eigen::Map<Eigen::ArrayXd>(par.data(),par.size());
    update_parameters(gammaa);
  }
  
  // double loglik(const Eigen::ArrayXd &u){
  //   double logl;
  //   double logdetD = 0;
  //   for (auto& k : chol_->D)
  //     logdetD += log(k);
  //   std::vector<double> v(u.data(), u.data()+u.size());// = arma::conv_to<std::vector<double>>::from(u);//std::vector<double> v(n_);
  //   chol_->ldl_lsolve(&v[0]);
  //   chol_->ldl_d2solve(&v[0]);
  //   double quadform = glmmr::algo::inner_sum(&v[0],&v[0],n_);
  //   logl = (-0.5*n_ * log(2*M_PI) - 0.5*logdetD - 0.5*quadform);
  //   return logl;
  // }
  
  double loglik(const Eigen::MatrixXd &u){
    int ncols = u.cols();
    double logdetD = 0;
    for (auto& k : chol_->D)
      logdetD += log(k);
    Eigen::ArrayXd logl(ncols);
    
#pragma omp parallel for
    for(int i = 0; i < ncols; i++){
      std::vector<double> v(u.data(), u.data()+u.size());// = arma::conv_to<std::vector<double>>::from(u);//std::vector<double> v(n_);
      chol_->ldl_lsolve(&v[0]);
      chol_->ldl_d2solve(&v[0]);
      double quadform = glmmr::algo::inner_sum(&v[0],&v[0],n_);
      logl(i) = (-0.5*n_ * log(2*M_PI) - 0.5*logdetD - 0.5*quadform);
    }
    
    return logl.mean();
  }
  
  ~SparseDMatrix(){
    delete mat_;
    delete chol_;
  }
};
}




#endif