#ifndef MHMCMC_H
#define MHMCMC_H

#include <RcppEigen.h>
#include <glmmr.h>
#include "moremaths.h"
#include "mcmlmodel.h"
#include <random>

// #ifdef _OPENMP
// #include <omp.h>
// #endif

namespace glmmr {

namespace mcmc {



class mcmcRunHMC {
public:
  glmmr::mcmlModel* model_;
  int trace_;
  Eigen::VectorXd u_;
  Eigen::VectorXd up_;
  Eigen::VectorXd r_;
  Eigen::VectorXd grad_;
  int refresh_;
  double lambda_;
  int max_steps_;
  std::minstd_rand gen_;
  std::uniform_real_distribution<double> dist_;
  int accept_;
  double e_;
  double ebar_;
  int steps_;
  double H_;
  double target_accept_;

  mcmcRunHMC(glmmr::mcmlModel* model, int trace = 0,
          double lambda = 0.01, int refresh = 500, int max_steps = 100,
          double target_accept = 0.9) : model_(model),
          trace_(trace),   u_(model_->Q_),up_(model_->Q_),r_(model_->Q_),
          grad_(model_->Q_), 
           refresh_(refresh),lambda_(lambda), max_steps_(max_steps),
           target_accept_(target_accept){
    initialise_u();
  }

  void initialise_u(){
    Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_);
    u_ = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z);
    z = Rcpp::rnorm(u_.size());
    r_ = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z);
    up_ = u_;
    accept_ = 0;
    H_ = 0;
    gen_ = std::minstd_rand(std::random_device{}());
    dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    e_ = 0.001;
    ebar_ = 1.0;
  }

  void new_proposal(bool adapt = false, int iter = 1){
    Rcpp::NumericVector z = Rcpp::rnorm(r_.size());
    r_ = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z);
    grad_ = model_->log_grad(u_);
    
    double lpr_ = 0.5*r_.transpose()*r_;
    up_ = u_;
    
    steps_ = std::max(1,(int)std::round(lambda_/e_));
    steps_ = std::min(steps_, max_steps_);
    
    // leapfrog integrator
    for(int i=0; i< steps_; i++){
      r_ += (e_/2)*grad_;
      up_ += e_ * r_;
      grad_ = model_->log_grad(up_);
      r_ += (e_/2)*grad_;
    }
    
    double lprt_ = 0.5*r_.transpose()*r_;

    double l1 = model_->log_prob(u_);
    double l2 = model_->log_prob(up_);
    double prob = std::min(1.0,exp(-l1 + lpr_ + l2 - lprt_));
    double runif = dist_(gen_); //(double)Rcpp::runif(1)(0);
    bool accept = runif < prob;
    
    if(trace_==2){
      Rcpp::Rcout << "\nIter: " << iter << " l1 " << l1 << " h1 " << lpr_ << " l2 " << l2 << " h2 " << lprt_;
      Rcpp::Rcout << "\nCurrent value: " << u_.transpose().head(10);
      Rcpp::Rcout << "\nvelocity: " << r_.transpose().head(10);
      Rcpp::Rcout << "\nProposal: " << up_.transpose().head(10);
      Rcpp::Rcout << "\nAccept prob: " << prob << " step size: " << e_ << " mean: " << ebar_ << " steps: " << steps_;
      if(accept){
        Rcpp::Rcout << " ACCEPT \n";
      } else {
        Rcpp::Rcout << " REJECT \n";
      }
    }
    
    
    if(accept){
      u_ = up_;
      accept_++;
    }
    
    if(adapt){
      double f1 = 1.0/(iter + 10);
      H_ = (1-f1)*H_ + f1*(target_accept_ - prob);
      double loge = -4.60517 - (sqrt((double)iter / 0.05))*H_;
      double powm = std::pow(iter,-0.75);
      double logbare = powm*loge + (1-powm)*log(ebar_);
      e_ = exp(loge);
      ebar_ = exp(logbare);
    } else {
      e_ = ebar_;
    }

    

    

  }

  Eigen::ArrayXXd sample(int warmup,
                         int nsamp,
                         int adapt = 100){
    int totalsamps = nsamp + warmup;
    int Q = model_->Q_;
    Eigen::MatrixXd samples(Q,nsamp+1);
    initialise_u();
    int i;

    // warmups
    for(i = 0; i < warmup; i++){
      if(i < adapt){
        new_proposal(true,i+1);
      } else {
        new_proposal(false);
      }
      if(i%refresh_== 0){
        Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
      }
    }

    samples.col(0) = u_;
    int iter = 1;
    //sampling
    for(i = 0; i < nsamp; i++){
      new_proposal(false);
      samples.col(i+1) = u_;
      if(i%refresh_== 0){
        Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
      }
    }
    if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(warmup+nsamp) << " steps: " << steps_ << " step size: " << e_;
    //return samples;
    return ((*(model_->L_)) * samples).array();

  }


};


}

}

#endif