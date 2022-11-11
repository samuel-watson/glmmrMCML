#ifndef MHMCMC_H
#define MHMCMC_H

#include <RcppEigen.h>
#include <glmmr.h>
#include "moremaths.h"

// a VERY basic metropolis hastings sampler for the random effects

namespace glmmr {

namespace mcmc {

class mcmcModel {
  public:
    const Eigen::MatrixXd &X_;
    Eigen::VectorXd xb_;
    Eigen::MatrixXd ZL_;
    Eigen::MatrixXd L_;
    const Eigen::VectorXd &y_;
    double var_par_;
    std::string family_; 
    std::string link_;
    int n;
    int Q;
    
    mcmcModel(
      const Eigen::MatrixXd &Z, 
      const Eigen::MatrixXd &L,
      const Eigen::MatrixXd &X,
      const Eigen::VectorXd &y, 
      const Eigen::VectorXd &beta,
      double var_par,
      std::string family, 
      std::string link
    ) : X_(X), xb_(X.rows()), ZL_(X.rows(),X.rows()), L_(L),  y_(y), var_par_(var_par), family_(family), link_(link) {
      xb_ = X*beta;
      ZL_ = Z*L;  
      n = X.rows();
      Q = Z.cols();
    }
    
    double log_prob(const Eigen::VectorXd &u){
      Eigen::VectorXd mu = xb_ + ZL_*u;
      Eigen::ArrayXd ll(n);
      Eigen::ArrayXd lp(Q);
#pragma omp parallel for
      for(int i = 0; i < n; i++){
        ll(i) = glmmr::maths::log_likelihood(y_(i),mu(i),var_par_,family_,link_);
      }
      
#pragma omp parallel for
      for(int i = 0; i < Q; i++){
        lp(Q) = glmmr::maths::log_likelihood(u(i),0,1,family_,link_);
      }
      
      return ll.sum() + lp.sum();
      
    }
    
    void update_beta(const Eigen::VectorXd &beta){
      xb_ = X_*beta;
    }
    
    // update this for all the gradient in glmmr maths
    Eigen::VectorXd log_grad(const Eigen::VectorXd &u){
      Eigen::VectorXd resid = y_ - xb_ - ZL_*u;
      Eigen::VectorXd grad = (1/var_par_)*ZL_.transpose()*resid + 0.5*u;
      return grad;
    }
    
};

class mcmcRun {
  public:
    glmmr::mcmc::mcmcModel* model_;
    
    mcmcRun(glmmr::mcmc::mcmcModel* model) : model_(model) {}
    
    Eigen::VectorXd new_proposal(const Eigen::VectorXd &u){
      Eigen::VectorXd prop(u.size());
      Eigen::VectorXd grad = model_->log_grad(u);
      for(int i = 0; i< u.size(); i++){
        prop(i) = Rcpp::rnorm(1,0.1*grad(i),0.2)(0);
      }
      return(prop);
    }
    
    Eigen::ArrayXXd sample(int warmup, 
                              int nsamp, 
                              int thin = 0){
      int thinv = thin <= 1 ? 1 : thin;
      int tsamps = (int)floor(nsamp/thinv);
      int totalsamps = nsamp + warmup;
      int Q = model_->Q;
      Eigen::MatrixXd samples(Q,tsamps+1);
      Eigen::VectorXd start(Q);
      Rcpp::NumericVector z = Rcpp::rnorm(Q);
      int i;
      for(i = 0; i < Q; i++) start(i) = z[i];
      
      // warmups
      for(i = 0; i < warmup; i++){
        Eigen::VectorXd proposal = new_proposal(start);
        double post = model_->log_prob(start);
        double postprop = model_->log_prob(proposal);
        double prob = exp(postprop - post);
        double runif = (double)Rcpp::runif(1)(0);
        if(runif < prob){
          start = proposal;
        } 
      }
      
      samples.col(0) = start;
      int iter = 1;
      //sampling
      if(thinv > 1){
        for(i = 0; i < nsamp; i++){
          Eigen::VectorXd proposal = new_proposal(start);
          double post = model_->log_prob(start);
          double postprop = model_->log_prob(proposal);
          double prob = exp(postprop - post);
          double runif = (double)Rcpp::runif(1)(0);
          if(runif < prob){
            start = proposal;
          }
          if(i%thinv == 0) {
            samples.col(iter) = start;
            iter++;
          }
        }
      } else {
        for(i = 0; i < nsamp; i++){
          Eigen::VectorXd proposal = new_proposal(samples.col(i));
          double post = model_->log_prob(samples.col(i));
          double postprop = model_->log_prob(proposal);
          double prob = exp(postprop - post);
          double runif = (double)Rcpp::runif(1)(0);
          if(runif < prob){
            samples.col(i+1) = proposal;
          } else {
            samples.col(i+1) = samples.col(i);
          }
        }
      }
      
      return (model_->L_.transpose()*samples).array();
      
    }
    
    
};


}

}

#endif