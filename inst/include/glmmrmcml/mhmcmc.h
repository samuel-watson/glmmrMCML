#ifndef MHMCMC_H
#define MHMCMC_H

#include <RcppEigen.h>
#include <glmmr.h>
#include "moremaths.h"
#include "mcmlmodel.h"

// #ifdef _OPENMP
// #include <omp.h>
// #endif

// a VERY basic metropolis hastings sampler for the random effects

namespace glmmr {

namespace mcmc {

class mcmcRun {
  public:
    glmmr::mcmlModel* model_;
    int trace_;
    Eigen::VectorXd u_;
    int accept_;
    double step_size_;
    int refresh_;
    
    mcmcRun(glmmr::mcmlModel* model, int trace = 0,
            double step_size = 0.015, int refresh = 500) : model_(model), trace_(trace),
      u_(model_->Q_), step_size_(step_size), refresh_(refresh) {
      initialise_u();
    }
    
    void initialise_u(){
      Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_);
      for(int i = 0; i < model_->Q_; i++) u_(i) = z[i];
      accept_ = 0;
    }
    
    void new_proposal(){
      
      Eigen::VectorXd prop(u_.size());
      Eigen::VectorXd prob_prop(u_.size());
      Eigen::VectorXd prob_u(u_.size());
      Eigen::VectorXd grad = model_->log_grad(u_);
      std::string family = model_->family_;
      std::string link = model_->link_;
      Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_,0,sqrt(2*step_size_));
      
      //get proposal probabilities
//#pragma omp parallel for
      // for(int i = 0; i< u_.size(); i++){
      //   prop(i) = Rcpp::rnorm(1, u_(i) + step_size_*grad(i),sqrt(2*step_size_))(0);
      //   // prob_prop(i) = glmmr::maths::log_likelihood(prop(i),u_(i) + step_size_*grad(i),
      //   //      sqrt(2*step_size_),"gaussian","identity");
      //   // double m1 = (prop(i)-u_(i) + step_size_*grad(i));
      //   // prob_prop(i) = (-1/(4*step_size_))*exp(m1*m1);
      // }
      
#pragma omp parallel for
      for(int i = 0; i< u_.size(); i++){
        prop(i) = u_(i) + step_size_*grad(i) + (double)z[i];
        // prob_prop(i) = glmmr::maths::log_likelihood(prop(i),u_(i) + step_size_*grad(i),
        //      sqrt(2*step_size_),"gaussian","identity");
        double m1 = (prop(i)-u_(i) - step_size_*grad(i));
        prob_prop(i) = (-1.0/(4*step_size_))*m1*m1;
      }
      
      grad = model_->log_grad(prop);
      
#pragma omp parallel for
      for(int i = 0; i< u_.size(); i++){
        // prob_u(i) = glmmr::maths::log_likelihood(u_(i),prop(i) + step_size_*grad(i),
        //           sqrt(2*step_size_),"gaussian","identity");
        double m1 = (u_(i) - prop(i) - step_size_*grad(i));
        prob_u(i) = (-1.0/(4*step_size_))*m1*m1;
      }
      
      double post = model_->log_prob(u_);
      double postprop = model_->log_prob(prop);
      double ppprop = prob_prop.sum();
      double ppu = prob_u.sum();
      //Rcpp::Rcout << "\n " << post << " " << postprop << " " << ppprop << " " << ppu;
      double prob = exp(postprop + ppu - post - ppprop);
      double runif = (double)Rcpp::runif(1)(0);
      bool accept = runif < prob;
      
      if(trace_==2){
        Rcpp::Rcout << "\nCurrent value: " << u_.transpose().head(10);
        Rcpp::Rcout << "\nGradient: " << grad.transpose().head(10);
        Rcpp::Rcout << "\nProposal: " << prop.transpose().head(10);
        Rcpp::Rcout << "\nAccept prob: " << prob << " random u: " << runif;
        if(accept){
          Rcpp::Rcout << " ACCEPT \n";
        } else {
          Rcpp::Rcout << " REJECT \n";
        }
      }
      
      if(accept){
        u_ = prop;
        accept_++;
      } 
      
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
      initialise_u();
      int i;
      
      // warmups
      for(i = 0; i < warmup; i++){
        new_proposal();
        if(i%refresh_== 0){
          Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
        }
      }
      
      samples.col(0) = u_;
      int iter = 1;
      //sampling
      if(thinv > 1){
        for(i = 0; i < nsamp; i++){
          new_proposal();
          if(i%thinv == 0) {
            samples.col(iter) = u_;
            iter++;
          }
          if(i%refresh_== 0){
            Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
          }
        }
      } else {
        for(i = 0; i < nsamp; i++){
          new_proposal();
          samples.col(i+1) = u_;
          if(i%refresh_== 0){
            Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
          }
        }
      }
      if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(warmup+nsamp);
      //return samples;
      return ((*(model_->L_)) * samples).array();
      
    }
    
    
};

class mcmcRunHMC {
public:
  glmmr::mcmlModel* model_;
  int trace_;
  Eigen::VectorXd u_;
  int accept_;
  double step_size_;
  int refresh_;
  int steps_;
  
  mcmcRunHMC(glmmr::mcmlModel* model, int trace = 0,
          double step_size = 0.015, int steps = 10, int refresh = 500) : model_(model), 
          trace_(trace), steps_(steps),
          u_(model_->Q_), step_size_(step_size), refresh_(refresh) {
    initialise_u();
  }
  
  void initialise_u(){
    Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_);
    for(int i = 0; i < model_->Q_; i++) u_(i) = z[i];
    accept_ = 0;
  }
  
  void new_proposal(){
    
    Eigen::VectorXd prop(u_.size());
    Eigen::VectorXd grad = model_->log_grad(u_);
    Eigen::VectorXd velocity(u_.size());
    //get proposal probabilities
    
    Rcpp::NumericVector z = Rcpp::rnorm(u_.size());
    for(int i = 0; i < u_.size(); i++) velocity(i) = z[i];
    
    double h1 = 0;
    for(int i = 0; i< u_.size(); i++){
      h1 += glmmr::maths::log_likelihood(velocity(i),0,1,"gaussian","identity");
    }
    
    prop = u_;
    
    for(int i=0; i< steps_; i++){
      velocity -= (step_size_)*grad;
      prop += step_size_ * velocity;
      //grad = model_->log_grad(prop);
      //velocity -= (step_size_/2)*grad;
    }
    
    double h2 = 0;
    for(int i = 0; i< u_.size(); i++){
      h2 += glmmr::maths::log_likelihood(velocity(i),0,1,"gaussian","identity");
    }
    
    double l1 = model_->log_prob(u_);
    double l2 = model_->log_prob(prop);
    
    double prob = exp(-l1 - h1 + l2 +h2);
    double runif = (double)Rcpp::runif(1)(0);
    bool accept = runif < prob;
    
    if(trace_==2){
      Rcpp::Rcout << "\n l1 " << l1 << " h1 " << h1 << " l2 " << l2 << " h2 " << h2;
      Rcpp::Rcout << "\nCurrent value: " << u_.transpose().head(10);
      Rcpp::Rcout << "\nProposal: " << prop.transpose().head(10);
      Rcpp::Rcout << "\nvelocity: " << velocity.transpose().head(10);
      Rcpp::Rcout << "\nAccept prob: " << prob << " random u: " << runif;
      if(accept){
        Rcpp::Rcout << " ACCEPT \n";
      } else {
        Rcpp::Rcout << " REJECT \n";
      }
    }
    
    if(accept){
      u_ = prop;
      accept_++;
    } 
    
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
    initialise_u();
    int i;
    
    // warmups
    for(i = 0; i < warmup; i++){
      new_proposal();
      if(i%refresh_== 0){
        Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
      }
    }
    
    samples.col(0) = u_;
    int iter = 1;
    //sampling
    if(thinv > 1){
      for(i = 0; i < nsamp; i++){
        new_proposal();
        if(i%thinv == 0) {
          samples.col(iter) = u_;
          iter++;
        }
        if(i%refresh_== 0){
          Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
        }
      }
    } else {
      for(i = 0; i < nsamp; i++){
        new_proposal();
        samples.col(i+1) = u_;
        if(i%refresh_== 0){
          Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
        }
      }
    }
    if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(warmup+nsamp);
    //return samples;
    return ((*(model_->L_)) * samples).array();
    
  }
  
  
};


}

}

#endif