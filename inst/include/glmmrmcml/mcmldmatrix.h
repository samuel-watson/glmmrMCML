#ifndef MCMLDMATRIX_H
#define MCMLDMATRIX_H

#define _USE_MATH_DEFINES

#include <cmath> 
#include <glmmr.h>
#include "moremaths.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
  class MCMLDmatrix: public DMatrix {
    public: 
      using DMatrix::DMatrix;
      
      double loglik(const Eigen::MatrixXd &u){
        int ncols = u.cols();
        double loglV = 0;
        for(int b=0;b<data_->B_;b++){
          Eigen::ArrayXd loglB = Eigen::ArrayXd::Zero(ncols);
          data_->subdata(b);
          int n = data_->n_dim();
          int m = data_->matstart_;
#pragma omp parallel for
          for(int i = 0; i < ncols; i++){
            Eigen::VectorXd uvals = u.col(i).segment(m, n);
            loglB(i) = loglik_block(b,uvals);
          }
          loglV += loglB.sum();
        }
        return loglV/ncols;
      }
      
      double logdet(){
        double logdetD = 0;
        for(int b=0;b<data_->B_;b++){
          Eigen::MatrixXd dmat = gen_block_mat(b,true,true);
          data_->subdata(b);
          int n = data_->n_dim();
          for(int i = 0; i < n; i++){
            logdetD += 2*log(dmat(i,i));
          }
        }
        return logdetD;
      }
      
    private:
      double loglik_block(int b, Eigen::VectorXd u){
        double logl = 0;
        Eigen::MatrixXd dmat = gen_block_mat(b,true,false);
        int n = u.size();
        if((data_->subcov_.col(2) == 1).all()){
          for(int k=0; k<n; k++){
            logl += -0.5*log(dmat(k,k)) -0.5*log(2*M_PI) -
              0.5*u(k)*u(k)/dmat(k,k);
          }
        } else {
          double logdetD = 0;
          for(int i = 0; i < n; i++){
            logdetD += 2*log(dmat(i,i));
          }
          Eigen::VectorXd zquad;
          double quadform = 0;
          zquad = glmmr::algo::forward_sub(&dmat,&u,n); 
          quadform = zquad.transpose()*zquad;
          logl = (-0.5*n * log(2*M_PI) - 0.5*logdetD - 0.5*quadform);
        }
        return logl;
      }
  };
}

#endif