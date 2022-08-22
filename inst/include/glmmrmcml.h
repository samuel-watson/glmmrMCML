#ifndef GLMMRMCML_H
#define GLMMRMCML_H

#include <cmath>  
#include <RcppArmadillo.h>
#include <rbobyqa.h>
#include "glmmr.h"
using namespace rminqa;

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

inline double log_factorial_approx(int n){
  double ans;
  if(n==0){
    ans = 0;
  } else {
    ans = n*log(n) - n + log(n*(1+4*n*(1+2*n)))/6 + log(arma::datum::pi)/2;
  }
  return ans;
}

inline double log_likelihood(arma::vec y,
                             arma::vec mu,
                             double var_par,
                             std::string family,
                             std::string link) {
  double logl = 0;
  arma::uword n = y.n_elem;
  const static std::unordered_map<std::string,int> string_to_case{
    {"poissonlog",1},
    {"poissonidentity",2},
    {"binomiallogit",3},
    {"binomiallog",4},
    {"binomialidentity",5},
    {"binomialprobit",6},
    {"gaussianidentity",7},
    {"gaussianlog",8}
  };
  switch (string_to_case.at(family+link)){
  case 1:
    for(arma::uword i=0;i<n; i++){
      double lf1 = log_factorial_approx(y[i]);
      logl += y(i)*mu(i) - exp(mu(i))-lf1;
    }
    break;
  case 2:
    for(arma::uword i=0;i<n; i++){
      double lf1 = log_factorial_approx(y[i]);
      logl += y(i)*log(mu(i)) - mu(i)-lf1;
    }
    break;
  case 3:
    for(arma::uword i=0; i<n; i++){
      if(y(i)==1){
        logl += log(1/(1+exp(-mu[i])));
      } else if(y(i)==0){
        logl += log(1 - 1/(1+exp(-mu[i])));
      }
    }
    break;
  case 4:
    for(arma::uword i=0; i<n; i++){
      if(y(i)==1){
        logl += mu(i);
      } else if(y(i)==0){
        logl += log(1 - exp(mu(i)));
      }
    }
    break;
  case 5:
    logl = 0;
    break;
  case 6:
    logl = 0;
    break;
  case 7:
    for(arma::uword i=0; i<n; i++){
      logl += -1*log(var_par) -0.5*log(2*arma::datum::pi) -
        0.5*pow((y(i) - mu(i))/var_par,2);
    }
    break;
  case 8:
    logl = 0;
  }
  
  return logl;
}

class D_likelihood : public Functor {
  DMatrix* D_;
  arma::mat u_;
  
public:
  D_likelihood(DMatrix* D,
               const arma::mat &u) : 
   D_(D), u_(u) {}
  double operator()(const vec &par) override{
    arma::uword nrow = u_.n_cols;
    D_->gamma_ = par;
    D_->gen_blocks_byfunc();
    double logl = 0;
//#pragma omp parallel for
    for(arma::uword j=0;j<nrow;j++){
      logl += D_->loglik(u_.col(j));
    }
    return -1*logl/nrow;
  }
};


class L_likelihood : public Functor {
  arma::mat Z_;
  arma::mat X_;
  arma::vec y_;
  arma::mat u_;
  std::string family_;
  std::string link_;
  
public:
  L_likelihood(const arma::mat &Z, 
               const arma::mat &X,
               const arma::vec &y, 
               const arma::mat &u, 
               std::string family, 
               std::string link) : 
  Z_(Z), X_(X), y_(y), u_(u),family_(family) , link_(link) {}
  double operator()(const vec &par) override{
    arma::uword niter = u_.n_cols;
    arma::uword n = y_.n_elem;
    arma::uword P = par.n_elem;
    arma::vec xb(n);
    double var_par;
    if(family_=="gaussian"){
      xb = X_*par.subvec(0,P-2);
      var_par = par(P-1);
    } else {
      xb = X_*par;
      var_par = 0;
    }
    
    arma::vec ll(niter,fill::zeros);
    arma::mat zd = Z_ * u_;
#pragma omp parallel for
    for(arma::uword j=0; j<niter ; j++){
      ll(j) += log_likelihood(y_,xb + zd.col(j),var_par,family_,link_);
    }
    
    return -1 * mean(ll);
  }
};

class F_likelihood : public Functor {
  DMatrix* D_;
  arma::mat Z_;
  arma::mat X_;
  arma::vec y_;
  arma::mat u_;
  arma::vec cov_par_fix_;
  std::string family_;
  std::string link_;
  bool importance_;
  
public:
  F_likelihood(DMatrix* D,
               arma::mat Z, 
               arma::mat X,
               arma::vec y, 
               arma::mat u,
               arma::vec cov_par_fix,
               std::string family, 
               std::string link,
               bool importance) : 
  D_(D) ,Z_(Z), X_(X), y_(y), 
  u_(u),cov_par_fix_(cov_par_fix), family_(family) , link_(link),
  importance_(importance) {}
  double operator()(const vec &par) override{
    
    arma::uword niter = u_.n_cols;
    arma::uword n = y_.n_elem;
    arma::uword P = X_.n_cols;
    arma::uword Q = cov_par_fix_.n_elem;
    double du;
    D_->gamma_ = par.subvec(P,P+Q-1);
    arma::field<arma::mat> Dfield = D_->genD();
    arma::vec numerD(niter,fill::zeros);
    double logdetD;
    arma::uword ndim_idx = 0;
    for(arma::uword b=0;b<D_->B_;b++){
      if(all(D_->func_def_.row(b)==1)){
#pragma omp parallel for collapse(2)
        for(arma::uword j=0;j<niter;j++){
          for(arma::uword k=0; k<Dfield[b].n_rows; k++){
            numerD(j) += -0.5*log(Dfield[b](k,k)) -0.5*log(2*arma::datum::pi) -
              0.5*pow(u_(ndim_idx+k,j),2)/Dfield[b](k,k);
          }
        }
        
      } else {
        arma::mat invD = inv_sympd(Dfield[b]);
        logdetD = arma::log_det_sympd(Dfield[b]);
#pragma omp parallel for
        for(arma::uword j=0;j<niter;j++){
          numerD(j) += log_mv_gaussian_pdf(u_.col(j).subvec(ndim_idx,ndim_idx+D_->N_dim_(b)-1),
                 invD,logdetD);
        }
      }
      ndim_idx += D_->N_dim_(b);
    }
    
    // log likelihood for observations
    //arma::vec zd(n);
    arma::vec xb(n);
    double var_par;
    
    if(family_=="gaussian"){
      var_par = par(P+Q);
    } else {
      var_par = 0;
    }
    
    xb = X_*par.subvec(0,P-1);
    arma::vec lfa(niter,fill::zeros);
    arma::mat zd = Z_ * u_;
#pragma omp parallel for
    for(arma::uword j=0; j<niter ; j++){
      lfa(j) += log_likelihood(y_,
          xb + zd.col(j),
          var_par,
          family_,
          link_);
    }
    
    if(importance_){
      // denominator density for importance sampling
      D_->gamma_ = cov_par_fix_;
      Dfield = D_->genD();
      arma::vec denomD(niter,fill::zeros);
      double logdetD;
      arma::uword ndim_idx = 0;
      for(arma::uword b=0;b<D_->B_;b++){
        if(all(D_->func_def_.row(b)==1)){
#pragma omp parallel for collapse(2)
          for(arma::uword j=0;j<niter;j++){
            for(arma::uword k=0; k<Dfield[b].n_rows; k++){
              denomD(j) += -0.5*log(Dfield[b](k,k)) -0.5*log(2*arma::datum::pi) -
                0.5*pow(u_(ndim_idx+k,j),2)/Dfield[b](k,k);
            }
          }
          
        } else {
          arma::mat invD = inv_sympd(Dfield[b]);
          logdetD = arma::log_det_sympd(Dfield[b]);
#pragma omp parallel for
          for(arma::uword j=0;j<niter;j++){
            denomD(j) += log_mv_gaussian_pdf(u_.col(j).subvec(ndim_idx,ndim_idx+D_->N_dim_(b)-1),
                   invD,logdetD);
          }
        }
        ndim_idx += D_->N_dim_(b);
      }
      du = 0;
      for(arma::uword j=0;j<niter;j++){
        du  += exp(lfa(j)+numerD(j))/exp(denomD(j));
      }
      
      du = -1 * log(du/niter);
    } else {
      du = -1* (mean(numerD) + mean(lfa));
    }
    
    return du;
  }
};

class mcmloptim{
public:
  mcmloptim(
    DMatrix* D,
    const arma::mat &Z, 
    const arma::mat &X,
    const arma::vec &y, 
    const arma::mat &u,
    const arma::vec &cov_par_fix,
    std::string family, 
    std::string link,
    const arma::vec &start,
    const arma::vec &lower_b,
    const arma::vec &upper_b,
    const arma::vec &lower_t,
    const arma::vec &upper_t,
    int trace
  ):
  D_(D),Z_(Z), X_(X), y_(y), 
  u_(u),cov_par_fix_(cov_par_fix), family_(family) , link_(link),
  start_(start), lower_b_(lower_b), upper_b_(upper_b), lower_t_(lower_t),
  upper_t_(upper_t), trace_(trace) {
    P_ = lower_b_.size();
    Q_ = lower_t_.size();
    beta_ = arma::vec(P_,fill::zeros);
    theta_ = arma::vec(Q_,fill::zeros);
    sigma_ = 0;
    n_ = y_.n_elem;
    niter_ = u_.n_cols;
  }
  
  DMatrix* D_;
  const arma::mat Z_;
  const arma::mat X_;
  const arma::vec y_;
  arma::mat u_;
  arma::vec cov_par_fix_;
  std::string family_;
  std::string link_;
  arma::vec start_;
  arma::vec lower_b_;
  arma::vec upper_b_;
  arma::vec lower_t_;
  arma::vec upper_t_;
  int trace_;
  arma::uword P_;
  arma::uword Q_;
  arma::vec beta_;
  arma::vec theta_;
  double sigma_;
  arma::uword n_;
  arma::uword niter_;
  
  void d_optim(){
    D_likelihood ddl(D_,u_);
    
    Rbobyqa<D_likelihood> opt;
    opt.set_upper(upper_t_);
    opt.set_lower(lower_t_);
    opt.control.iprint = trace_;
    arma::vec start_t = start_.subvec(P_,P_+Q_-1);
    opt.minimize(ddl, start_t);
    theta_ = opt.par();
  }
  
  void l_optim(){
    L_likelihood ldl(Z_,X_,y_,u_,family_,link_);
    
    Rbobyqa<L_likelihood> opt;
    opt.set_upper(upper_b_);
    opt.set_lower(lower_b_);
    opt.control.iprint = trace_;
    arma::vec start_b = start_.subvec(0,P_-1);
    opt.minimize(ldl, start_b);
    beta_ = opt.par();
  }
  
  void f_optim(){
    F_likelihood dl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,true);
    
    Rbobyqa<F_likelihood> opt;
    opt.set_lower(arma::join_cols(lower_b_,lower_t_));
    opt.control.iprint = trace_;
    arma::vec allpars = arma::join_cols(beta_,theta_);
    opt.minimize(dl, allpars);
    arma::vec pars = opt.par();
    beta_ = pars(0,P_-1);
    theta_ = pars(P_,P_+Q_-1);
  }
  
  void mcnr(){
    //generate residuals
    arma::vec xb = X_*start_.subvec(0,X_.n_cols-1);
    arma::vec sigmas(niter_);
    arma::mat XtWX(P_,P_,fill::zeros);
    arma::vec Wu(n_,fill::zeros);
    arma::vec zd(n_,fill::zeros);
    arma::vec resid(n_,fill::zeros);
    arma::vec wdiag(n_,fill::zeros);
    arma::vec wdiag2(n_,fill::zeros);
    arma::mat W(n_,n_,fill::zeros);
    arma::mat W2(n_,n_,fill::zeros);
    for(arma::uword i = 0; i < niter_; ++i){
      zd = Z_ * u_.col(i);
      zd = mod_inv_func(xb + zd, link_);
      resid = y_ - zd;
      sigmas(i) = arma::stddev(resid);
      wdiag = dhdmu(xb + zd,family_,link_);
      wdiag2 = 1/arma::pow(wdiag, 2);
      if(family_=="gaussian" || family_=="gamma") wdiag2 *= sigmas(i);
      for(arma::uword j = 0; j<n_; j++){
        XtWX += wdiag2(j)*(X_.row(j).t()*X_.row(j));
        Wu(j) += wdiag(j)*wdiag2(j)*resid(j); 
      }
      //sigmas(i) = pow(sigmas(i),2);
    }
    //Rcpp::Rcout<< XtWX;
    XtWX = arma::inv_sympd(XtWX/niter_);
    Wu = Wu/niter_;
    
    arma::vec bincr = XtWX*X_.t()*Wu;
    beta_ = start_.subvec(0,P_-1) + bincr;
    sigma_ = mean(sigmas);
  }
  
  arma::mat b_hessian(double tol = 1e-4){
    L_likelihood hdl(Z_,X_,y_,u_,family_,link_);
    
    hdl.os.usebounds_ = 1;
    hdl.os.lower_ = lower_b_;
    hdl.os.upper_ = upper_b_;
    hdl.os.ndeps_ = arma::ones<arma::vec>(P_) * tol;
    arma::mat hessian(P_,P_,fill::zeros);
    arma::vec start_b = start_.subvec(0,P_-1);
    hdl.Hessian(start_b,hessian);
    return hessian;
  }
  
  arma::vec f_grad(double tol = 1e-4){
    F_likelihood fdl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,false);
    fdl.os.usebounds_ = 1;
    fdl.os.lower_ = arma::join_cols(lower_b_,lower_t_);
    fdl.os.upper_ = arma::join_cols(upper_b_,upper_t_);
    fdl.os.ndeps_ = arma::ones<arma::vec>(start_.size()) * tol;
    arma::vec gradient(start_.n_elem,fill::zeros);
    fdl.Gradient(start_,gradient);
    return gradient;
  }
  
  arma::mat f_hess(double tol = 1e-4){
    F_likelihood fhdl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,false);
    fhdl.os.usebounds_ = 1;
    fhdl.os.lower_ = arma::join_cols(lower_b_,lower_t_);
    fhdl.os.upper_ = arma::join_cols(upper_b_,upper_t_);
    fhdl.os.ndeps_ = arma::ones<arma::vec>(start_.size()) * tol;
    arma::mat hessian(start_.n_elem,start_.n_elem,fill::zeros);
    fhdl.Hessian(start_,hessian);
    return hessian;
  }
  
  arma::vec get_beta(){
    return beta_;
  }
  
  arma::vec get_theta(){
    return theta_;
  }
  
  double get_sigma(){
    return sigma_;
  }
  
};



#endif