#ifndef MHMCMC_H
#define MHMCMC_H

#include <RcppEigen.h>
#include <glmmr.h>
#include "moremaths.h"
#include "mcmlmodel.h"

// a VERY basic metropolis hastings sampler for the random effects

namespace glmmr {

namespace mcmc {

class mcmcRun {
  public:
    glmmr::mcmlModel* model_;
    int trace_;
    
    mcmcRun(glmmr::mcmlModel* model, int trace = 0) : model_(model), trace_(trace) {}
    
    Eigen::VectorXd new_proposal(const Eigen::VectorXd &u,
                                 double step_size = 0.015){
      Eigen::VectorXd prop(u.size());
      Eigen::VectorXd grad = model_->log_grad(u);
      
      for(int i = 0; i< u.size(); i++){
        prop(i) = Rcpp::rnorm(1, u(i) - 0.5*step_size*grad(i),sqrt(step_size))(0);
      }
      
      if(trace_==2){
        Rcpp::Rcout << "\nCurrent value: " << u.transpose().head(10);
        Rcpp::Rcout << "\nGradient: " << grad.transpose().head(10);
        Rcpp::Rcout << "\nProposal: " << prop.transpose().head(10);
      }
      return(prop);
    }
    
    Eigen::ArrayXXd sample(int warmup, 
                              int nsamp, 
                              int thin = 0,
                              double step_size = 0.015){
      int thinv = thin <= 1 ? 1 : thin;
      int tsamps = (int)floor(nsamp/thinv);
      int totalsamps = nsamp + warmup;
      int Q = model_->Q_;
      Eigen::MatrixXd samples(Q,tsamps+1);
      Eigen::VectorXd start(Q);
      Rcpp::NumericVector z = Rcpp::rnorm(Q);
      int i;
      for(i = 0; i < Q; i++) start(i) = z[i];
      int accept = 0;
      // warmups
      for(i = 0; i < warmup; i++){
        Eigen::VectorXd proposal = new_proposal(start,step_size);
        double post = model_->log_prob(start);
        double postprop = model_->log_prob(proposal);
        double prob = exp(postprop - post);
        double runif = (double)Rcpp::runif(1)(0);
        if(runif < prob){
          start = proposal;
          accept++;
        } 
      }
      
      samples.col(0) = start;
      int iter = 1;
      //sampling
      if(thinv > 1){
        for(i = 0; i < nsamp; i++){
          Eigen::VectorXd proposal = new_proposal(start,step_size);
          double post = model_->log_prob(start);
          double postprop = model_->log_prob(proposal);
          double prob = exp(postprop - post);
          double runif = (double)Rcpp::runif(1)(0);
          if(runif < prob){
            start = proposal;
            accept++;
          }
          if(i%thinv == 0) {
            samples.col(iter) = start;
            iter++;
          }
        }
      } else {
        for(i = 0; i < nsamp; i++){
          Eigen::VectorXd proposal = new_proposal(samples.col(i),step_size);
          double post = model_->log_prob(samples.col(i));
          double postprop = model_->log_prob(proposal);
          double prob = exp(postprop - post);
          double runif = (double)Rcpp::runif(1)(0);
          if(runif < prob){
            samples.col(i+1) = proposal;
            accept++;
          } else {
            samples.col(i+1) = samples.col(i);
          }
        }
      }
      if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept/(warmup+nsamp);
      //return samples;
      return ((*(model_->L_))*samples).array();
      
    }
    
    
};


}

}

#endif