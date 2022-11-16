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
    //Eigen::MatrixXd samp_cov_;
    //Eigen::MatrixXd samp_cov_L_;
    //int part_accept_;
    //Eigen::MatrixXd cache_samps_;
    //int cache_iter_;
    
    mcmcRun(glmmr::mcmlModel* model, int trace = 0,
            double step_size = 0.015, int refresh = 500) : model_(model), trace_(trace),
      u_(model_->Q_), step_size_(step_size), refresh_(refresh) {
      //samp_cov_(model_->Q_, model_->Q_), samp_cov_L_(model_->Q_, model_->Q_),
      //cache_samps_(model_->Q_,50) {
      initialise_u();
    }
    
    void initialise_u(){
      Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_);
      for(int i = 0; i < model_->Q_; i++) u_(i) = z[i];
      accept_ = 0;
      //part_accept_=0;
      //samp_cov_ = Eigen::MatrixXd::Identity(model_->Q_,model_->Q_);
      //samp_cov_L_ = samp_cov_;
      //cache_iter_ = 0;
    }
    
    // double logl_mvn(Eigen::VectorXd u){
    //   int n = model_->Q_;
    //   Eigen::VectorXd zquad;
    //   double quadform = 0;
    //   zquad = glmmr::algo::forward_sub(&samp_cov_L_,&u,n); 
    //   quadform = zquad.transpose()*zquad;
    //   return  -0.5*quadform;
    // }
    
    void new_proposal(bool adapt, int iter=1){
      
      Eigen::VectorXd prop(u_.size());
      //Eigen::VectorXd prob_prop(u_.size());
      //Eigen::VectorXd prob_u(u_.size());
      Eigen::VectorXd grad = model_->log_grad(u_);
      std::string family = model_->family_;
      std::string link = model_->link_;
      Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_,0,sqrt(2*step_size_));
      Eigen::Map<Eigen::VectorXd> zs(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z));
      
      Eigen::VectorXd mu = u_ +  step_size_*grad;
      //prop = mu + samp_cov_L_*zs;
      prop = mu + zs;
      //double ppprop = logl_mvn(prop - mu);
      double ppprop = (-1.0/(4*step_size_))*(prop - mu).transpose()*(prop - mu);
      
      
// #pragma omp parallel for
//       for(int i = 0; i< u_.size(); i++){
//         //prop(i) = u_(i) + step_size_*grad(i) + (double)z[i];
//         double m1 = (prop(i)-u_(i) - step_size_*grad(i));
//         prob_prop(i) = (-1.0/(4*step_size_))*m1*m1;
//       }
      
      grad = model_->log_grad(prop);
      //mu = prop +  step_size_*samp_cov_*grad;
      mu = prop +  step_size_*grad;
      //double ppu = logl_mvn(u_ - mu);
      double ppu = (-1.0/(4*step_size_))*(u_ - mu).transpose()*(u_ - mu);
      
// #pragma omp parallel for
//       for(int i = 0; i< u_.size(); i++){
//         double m1 = (u_(i) - prop(i) - step_size_*grad(i));
//         prob_u(i) = (-1.0/(4*step_size_))*m1*m1;
//       }
      
      double post = model_->log_prob(u_);
      double postprop = model_->log_prob(prop);
      //double ppprop = prob_prop.sum();
      //double ppu = prob_u.sum();
      double prob = exp(postprop + ppu - post - ppprop);
      prob = std::min(1.0, prob);
      double runif = (double)Rcpp::runif(1)(0);
      bool accept = runif < prob;
      
      // adaptation step
      if(adapt && iter > 0){
        double gam1 = pow(iter,-0.8);
        step_size_ += gam1*(prob - 0.574);
        if(step_size_ < 1e-6)step_size_ = 1e-6;
      }
      
      if(trace_==2){
        Rcpp::Rcout << "\nIter: " << iter << " step size: " << step_size_;
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
        //part_accept_++;
      } 
      
      
      
      
      // if(adapt){
      //   double acceptrate = (double)(part_accept_)/50;
      //   part_accept_ = 0;
      //   double gam1 = pow(iter,-0.6);
      //   double logsig = log(step_size_) + gam1*(acceptrate - 0.574);
      //   if(trace_>0)Rcpp::Rcout << "\nAdapt step size. Block accept rate: " << acceptrate << " Old: " << step_size_;
      //   step_size_ = exp(logsig);
      //   if(trace_>0)Rcpp::Rcout << " New: " << step_size_; 
      //   // now do covariance of samples
      //   //Eigen::VectorXd meanx = cache_samps_.rowwise().mean();
      //   //cache_samps_.colwise() -= meanx;
      //   //cache_iter_ = 0;
      //   //if(trace_>0) Rcpp::Rcout << "\nsamps:\n" << cache_samps_.block(0,0,10,10);
      //   //Eigen::MatrixXd samp_cov_hat_ = (cache_samps_*cache_samps_.transpose());
      //   //samp_cov_hat_ *= 0.02040816;
      //   //samp_cov_ += gam1*(samp_cov_hat_ - samp_cov_);
      //   //if(trace_>0) Rcpp::Rcout << "\ncov:\n" << samp_cov_hat_.block(0,0,10,10);
      //   //samp_cov_L_ = samp_cov_.llt().matrixL();
      //   //if(trace_>0) Rcpp::Rcout << "\nL:\n" << samp_cov_L_.block(0,0,10,10);
      // }
      
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
      bool adapt = false;
      
      // warmups
      for(i = 0; i < warmup; i++){
        //adapt = i%50 == 0 && i > 0;
        new_proposal(true,i);
        //cache_samps_.col(cache_iter_) = u_;
        //cache_iter_++;
        if(i%refresh_== 0){
          Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
        }
      }
      
      samples.col(0) = u_;
      int iter = 1;
      accept_ = 0;
      //sampling
      if(thinv > 1){
        for(i = 0; i < nsamp; i++){
          new_proposal(false);
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
          new_proposal(false);
          samples.col(i+1) = u_;
          if(i%refresh_== 0){
            Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
          }
        }
      }
      if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(nsamp) << " step size: " << step_size_;
      //return samples;
      return ((*(model_->L_)) * samples).array();
      
    }
    
    
};

// class mcmcRunHMC {
// public:
//   glmmr::mcmlModel* model_;
//   int trace_;
//   Eigen::VectorXd u_;
//   int accept_;
//   double step_size_;
//   int refresh_;
//   int steps_;
//   
//   mcmcRunHMC(glmmr::mcmlModel* model, int trace = 0,
//           double step_size = 0.015, int steps = 10, int refresh = 500) : model_(model), 
//           trace_(trace), steps_(steps),
//           u_(model_->Q_), step_size_(step_size), refresh_(refresh) {
//     initialise_u();
//   }
//   
//   void initialise_u(){
//     Rcpp::NumericVector z = Rcpp::rnorm(model_->Q_);
//     for(int i = 0; i < model_->Q_; i++) u_(i) = z[i];
//     accept_ = 0;
//   }
//   
//   void new_proposal(){
//     
//     Eigen::VectorXd prop(u_.size());
//     Eigen::VectorXd grad = model_->log_grad(u_);
//     Eigen::VectorXd velocity(u_.size());
//     //get proposal probabilities
//     
//     Rcpp::NumericVector z = Rcpp::rnorm(u_.size());
//     for(int i = 0; i < u_.size(); i++) velocity(i) = z[i];
//     
//     double h1 = 0;
//     for(int i = 0; i< u_.size(); i++){
//       h1 += glmmr::maths::log_likelihood(velocity(i),0,1,"gaussian","identity");
//     }
//     
//     prop = u_;
//     
//     for(int i=0; i< steps_; i++){
//       velocity -= (step_size_)*grad;
//       prop += step_size_ * velocity;
//       //grad = model_->log_grad(prop);
//       //velocity -= (step_size_/2)*grad;
//     }
//     
//     double h2 = 0;
//     for(int i = 0; i< u_.size(); i++){
//       h2 += glmmr::maths::log_likelihood(velocity(i),0,1,"gaussian","identity");
//     }
//     
//     double l1 = model_->log_prob(u_);
//     double l2 = model_->log_prob(prop);
//     
//     double prob = exp(-l1 - h1 + l2 +h2);
//     double runif = (double)Rcpp::runif(1)(0);
//     bool accept = runif < prob;
//     
//     if(trace_==2){
//       Rcpp::Rcout << "\n l1 " << l1 << " h1 " << h1 << " l2 " << l2 << " h2 " << h2;
//       Rcpp::Rcout << "\nCurrent value: " << u_.transpose().head(10);
//       Rcpp::Rcout << "\nProposal: " << prop.transpose().head(10);
//       Rcpp::Rcout << "\nvelocity: " << velocity.transpose().head(10);
//       Rcpp::Rcout << "\nAccept prob: " << prob << " random u: " << runif;
//       if(accept){
//         Rcpp::Rcout << " ACCEPT \n";
//       } else {
//         Rcpp::Rcout << " REJECT \n";
//       }
//     }
//     
//     if(accept){
//       u_ = prop;
//       accept_++;
//     } 
//     
//   }
//   
//   Eigen::ArrayXXd sample(int warmup, 
//                          int nsamp, 
//                          int thin = 0,
//                          double step_size = 0.015){
//     int thinv = thin <= 1 ? 1 : thin;
//     int tsamps = (int)floor(nsamp/thinv);
//     int totalsamps = nsamp + warmup;
//     int Q = model_->Q_;
//     Eigen::MatrixXd samples(Q,tsamps+1);
//     initialise_u();
//     int i;
//     
//     // warmups
//     for(i = 0; i < warmup; i++){
//       new_proposal();
//       if(i%refresh_== 0){
//         Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
//       }
//     }
//     
//     samples.col(0) = u_;
//     int iter = 1;
//     //sampling
//     if(thinv > 1){
//       for(i = 0; i < nsamp; i++){
//         new_proposal();
//         if(i%thinv == 0) {
//           samples.col(iter) = u_;
//           iter++;
//         }
//         if(i%refresh_== 0){
//           Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
//         }
//       }
//     } else {
//       for(i = 0; i < nsamp; i++){
//         new_proposal();
//         samples.col(i+1) = u_;
//         if(i%refresh_== 0){
//           Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
//         }
//       }
//     }
//     if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(warmup+nsamp);
//     //return samples;
//     return ((*(model_->L_)) * samples).array();
//     
//   }
//   
//   
// };


}

}

#endif