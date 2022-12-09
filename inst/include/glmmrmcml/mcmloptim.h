#ifndef MCMLCLASS_H
#define MCMLCLASS_H

#include <rbobyqa.h>
#include "likelihood.h"
#include <RcppEigen.h>
#include <glmmr/maths.h>
#include "mcmlmodel.h"

using namespace rminqa;

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {

template<typename T>
class mcmloptim{
public:
  mcmloptim(
    T* D,
    glmmr::mcmlModel* M,
    const Eigen::ArrayXd &start,
    int trace
  ):
  D_(D), M_(M), start_(start), trace_(trace) {
    Q_ = D->data_->n_cov_pars();
    cov_par_fix_ = start.segment(M_->P_,Q_);
    beta_ = start.segment(0,M_->P_); 
    theta_ = start.segment(M_->P_,Q_);
    sigma_ = M_->family_=="gaussian" ? start(M_->P_+Q_) : 0;
    for(int i = 0; i< M_->P_; i++){
      lower_b_.push_back(R_NegInf);
      upper_b_.push_back(R_PosInf);
    }
    for(int i = 0; i< Q_; i++){
      lower_t_.push_back(1e-6);
      upper_t_.push_back(R_PosInf);
    }
  }
  
  T* D_;
  glmmr::mcmlModel* M_;
  int Q_;
  Eigen::ArrayXd cov_par_fix_;
  Eigen::ArrayXd start_;
  int trace_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd theta_;
  double sigma_;
  std::vector<double> lower_b_;
  std::vector<double> upper_b_;
  std::vector<double> lower_t_;
  std::vector<double> upper_t_;
  
  // optimise multivariate normal distribution of random effects
  void d_optim(){
    glmmr::likelihood::D_likelihood<T> ddl(D_,M_->u_);
    Rbobyqa<glmmr::likelihood::D_likelihood<T>,std::vector<double> > opt;
    opt.set_upper(upper_t_);
    opt.set_lower(lower_t_);
    opt.control.iprint = trace_;
    std::vector<double> start_t;
    start_t.resize(Q_);
    Eigen::VectorXd::Map(&start_t[0], theta_.size()) = theta_;
    opt.minimize(ddl, start_t);
    std::vector<double> theta = opt.par();
    theta_ = Eigen::Map<Eigen::ArrayXd>(theta.data(),Q_);
  }
  
  // optimise likelihood conditional on random effects
  void l_optim(){
    glmmr::likelihood::L_likelihood ldl(M_);
    Rbobyqa<glmmr::likelihood::L_likelihood,std::vector<double> > opt;
    opt.control.iprint = trace_;
    std::vector<double> start_b;
    start_b.resize(M_->P_);
    Eigen::VectorXd::Map(&start_b[0], beta_.size()) = beta_;
    std::vector<double> lower_b = lower_b_;
    if(M_->family_=="gaussian"){
      start_b.push_back(sigma_);
      lower_b.push_back(0.0);
    }
    opt.set_lower(lower_b);
    opt.minimize(ldl, start_b);
    std::vector<double> beta = opt.par();
    beta_ = Eigen::Map<Eigen::ArrayXd>(beta.data(),M_->P_);
    if(M_->family_=="gaussian") sigma_ = beta[M_->P_];
  }
  
  // jointly optimise marginal y and random effects models
  void f_optim(){
    glmmr::likelihood::F_likelihood<T> dl(D_,M_,cov_par_fix_,true,true,sigma_);
    Rbobyqa<glmmr::likelihood::F_likelihood<T>, std::vector<double> > opt;
    std::vector<double> lower = lower_b_;
    std::vector<double> allpars;
    allpars.resize(M_->P_);
    Eigen::VectorXd::Map(&allpars[0], beta_.size()) = beta_;
    for(int i=0; i< theta_.size(); i++) {
      allpars.push_back(theta_(i));
      lower.push_back(lower_t_[i]);
    }
    if(M_->family_=="gaussian"){
      lower.push_back(0.0);
      allpars.push_back(sigma_);
    }
    opt.set_lower(lower);
    opt.control.iprint = trace_;
    opt.minimize(dl, allpars);
    std::vector<double> beta = opt.par();
    beta_ = Eigen::Map<Eigen::ArrayXd>(beta.data(),M_->P_);//beta_ = pars(0,P_-1);
    theta_ = Eigen::Map<Eigen::ArrayXd>(beta.data()+M_->P_,Q_);//theta_ = pars(P_,P_+Q_-1);
    if(M_->family_=="gaussian") sigma_ = beta[M_->P_+Q_];
  }
  
  //laplace approximation
  void la_optim(){
    glmmr::likelihood::LA_likelihood<T> ldl(M_,D_);
    Rbobyqa<glmmr::likelihood::LA_likelihood<T>,std::vector<double> > opt;
    opt.control.iprint = trace_;
    std::vector<double> start_b;
    start_b.resize(M_->P_ + M_->Q_);
    for(int i = 0; i< M_->P_; i++)start_b[i] = beta_(i);
    for(int i = 0; i< M_->Q_; i++)start_b[M_->P_ + i] = (*(M_->u_))(i,0);
    opt.minimize(ldl, start_b);
    std::vector<double> beta = opt.par();
    beta_ = Eigen::Map<Eigen::ArrayXd>(beta.data(),M_->P_);
    for(int i = 0; i< M_->Q_; i++)(*(M_->u_))(i,0) = beta[M_->P_ + i];
  }
  
  void la_optim_cov(){
    //Eigen::MatrixXd W = M_->genW();
    glmmr::likelihood::LA_likelihood_cov<T> ldl(M_,D_);//M_->L_
    Rbobyqa<glmmr::likelihood::LA_likelihood_cov<T>,std::vector<double> > opt;
    
    std::vector<double> lower = lower_t_;
    opt.control.iprint = trace_;
    std::vector<double> start_t;
    start_t.resize(Q_);
    Eigen::VectorXd::Map(&start_t[0], theta_.size()) = theta_;
    if(M_->family_=="gaussian"||M_->family_=="Gamma"||M_->family_=="beta"){
      lower.push_back(0.0);
      start_t.push_back(sigma_);
    }
    
    opt.set_lower(lower);
    opt.minimize(ldl, start_t);
    std::vector<double> theta = opt.par();
    theta_ = Eigen::Map<Eigen::ArrayXd>(theta.data(),Q_);
    if(M_->family_=="gaussian"||M_->family_=="Gamma"||M_->family_=="beta") sigma_ = theta[Q_];
  }
  
  void la_optim_bcov(){
    glmmr::likelihood::LA_likelihood_btheta<T> ldl(M_,D_);
    Rbobyqa<glmmr::likelihood::LA_likelihood_btheta<T>,std::vector<double> > opt;
    
    std::vector<double> lower = lower_b_;
    std::vector<double> allpars;
    allpars.resize(M_->P_);
    Eigen::VectorXd::Map(&allpars[0], beta_.size()) = beta_;
    for(int i=0; i< theta_.size(); i++) {
      allpars.push_back(theta_(i));
      lower.push_back(lower_t_[i]);
    }
    if(M_->family_=="gaussian"){
      lower.push_back(0.0);
      allpars.push_back(sigma_);
    }
    opt.set_lower(lower);
    opt.control.iprint = trace_;
    
    opt.minimize(ldl, allpars);
    
    std::vector<double> beta = opt.par();
    beta_ = Eigen::Map<Eigen::ArrayXd>(beta.data(),M_->P_);//beta_ = pars(0,P_-1);
    theta_ = Eigen::Map<Eigen::ArrayXd>(beta.data()+M_->P_,Q_);//theta_ = pars(P_,P_+Q_-1);
    if(M_->family_=="gaussian"||M_->family_=="Gamma"||M_->family_=="beta") sigma_ = beta[M_->P_+Q_];
  }
  
  Eigen::MatrixXd hess_la(double tol = 1e-4){
    glmmr::likelihood::LA_likelihood_btheta<T> hdl(M_,D_);
    int nvar = M_->P_ + theta_.size();
    if(M_->family_=="gaussian"||M_->family_=="Gamma"||M_->family_=="beta")nvar++;
    std::vector<double> ndep;
    
    for(int i = 0; i < nvar; i++) ndep.push_back(tol);
    hdl.os.ndeps_ = ndep;
    std::vector<double> hessian(nvar * nvar,0.0);
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    for(int i = 0; i < theta_.size(); i++) start_b.push_back(theta_(i));
    if(M_->family_=="gaussian"||M_->family_=="Gamma"||M_->family_=="beta")start_b.push_back(sigma_);
    hdl.Hessian(start_b,hessian);
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),nvar,nvar);
    return hess;
  }
  
  // newton raphson step
  void mcnr(){
    Eigen::ArrayXd sigmas(M_->niter_);
    int P = M_->P_;
    int niter = M_->niter_;
    int n = M_->n_;
    Eigen::MatrixXd XtXW = Eigen::MatrixXd::Zero(P*niter,P);
    Eigen::MatrixXd Wu = Eigen::MatrixXd::Zero(n,niter);
    //check if parallelisation is faster or not! if not then declare vectors in advance
 
    Eigen::MatrixXd zd = M_->get_zu();//M_->Z_ * (*(M_->u_));
    Eigen::VectorXd xb = M_->xb_;
    //test if parallelisation improves speed here
#pragma omp parallel for
    for(int i = 0; i < niter; ++i){
      //Eigen::VectorXd zd = Z_ * u_.col(i);
      M_->update_W(i);
      Eigen::VectorXd zdu = glmmr::maths::mod_inv_func(xb + zd.col(i), M_->link_);
      Eigen::ArrayXd resid = (M_->y_ - zdu).array();
      sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
      XtXW.block(P*i, 0, P, P) = M_->X_.transpose() * M_->W_ * M_->X_;
      Eigen::VectorXd dmu = glmmr::maths::detadmu(xb + zd.col(i),M_->link_);
      // Eigen::VectorXd wdiag = glmmr::maths::dhdmu(xb + zd.col(i),M_->family_,M_->link_);
      for(int j=0; j<n; j++){
        // double var = (M_->family_=="gaussian" || M_->family_=="Gamma"||M_->family_=="beta") ? sigmas(i) : 1.0;
        //XtXW.block(P*i, 0, P, P) +=  (var/(wdiag(j)*wdiag(j)))*((M_->X_.row(j).transpose()) * (M_->X_.row(j)));
        Wu(j,i) += M_->W_(j,j)*dmu(j)*resid(j);
        //Wu(j,i) += (var/wdiag(j))*resid(j);
      }
    }
    XtXW *= (double)1/niter;
    Eigen::MatrixXd XtWXm = XtXW.block(0,0,P,P);
    for(int i = 1; i<niter; i++) XtWXm += XtXW.block(P*i,0,P,P);
    XtWXm = XtWXm.inverse();
    Eigen::VectorXd Wum = Wu.rowwise().mean();
    Eigen::VectorXd bincr = XtWXm * (M_->X_.transpose()) * Wum;
    //Rcpp::Rcout << "\nbincr:\n" << bincr.transpose();
    beta_ += bincr;
    sigma_ = sigmas.mean();
  }
  
  void mcnr_b(){
    double sigmas;
    int P = M_->P_;
    int n = M_->n_;
    Eigen::VectorXd Wu = Eigen::VectorXd::Zero(n);
    //Eigen::MatrixXd zd = M_->zu_;
    Eigen::VectorXd zd = M_->ZL_ * M_->u_->col(0);
    Eigen::VectorXd xb = M_->xb_;
    Eigen::VectorXd dmu = glmmr::maths::detadmu(xb + zd,M_->link_);
    
    Eigen::MatrixXd LZWZL = M_->ZL_.transpose() * M_->W_ * M_->ZL_;
    //Eigen::MatrixXd I = Eigen::MatrixXd::Identity(LZWZL.rows(),LZWZL.cols());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(M_->Q_,M_->Q_);
    LZWZL.noalias() += I;//M_->D_.llt().solve(I);
    Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(LZWZL.rows(),LZWZL.cols());
    LZWZL = LZWZL.llt().solve(I2);
    
    Eigen::VectorXd zdu = glmmr::maths::mod_inv_func(xb + zd, M_->link_);
    Eigen::ArrayXd resid = (M_->y_ - zdu).array();
    sigmas = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
    
    
    for(int j=0; j<n; j++){
      Wu(j) += M_->W_(j,j)*dmu(j)*resid(j);
    }
    
    Eigen::MatrixXd XtXW = M_->X_.transpose() * M_->W_ * M_->X_;
    
    XtXW = XtXW.inverse();
    Eigen::VectorXd bincr = XtXW * (M_->X_).transpose() * Wu;
    Eigen::VectorXd vgrad = M_->log_grad(M_->u_->col(0),false);
    Eigen::VectorXd vincr = LZWZL * vgrad;

    // Eigen::MatrixXd fisher(LZWZL.rows() +XtXW.rows(), LZWZL.cols() +XtXW.cols() );
    // fisher.block(0,0,XtXW.rows(),XtXW.cols()) = XtXW;
    // fisher.block(XtXW.rows(),0,LZWZL.rows(),XtXW.cols()) = M_->Z_.transpose() * M_->W_ * M_->X_;
    // fisher.block(0,XtXW.cols(),XtXW.rows(),LZWZL.cols()) = M_->X_.transpose() * M_->W_ * M_->Z_ * M_->D_;
    // fisher.block(XtXW.rows(),XtXW.cols(),LZWZL.rows(),LZWZL.cols()) = LZWZL;
    // 
    // Eigen::VectorXd fisher2(LZWZL.rows() +XtXW.rows());
    // fisher2.segment(0,XtXW.rows()) = (M_->X_.transpose()) * Wu;
    // fisher2.segment(XtXW.rows(),LZWZL.rows()) = M_->log_grad(M_->u_->col(0));
    //  
    // Eigen::VectorXd solu = fisher.llt().solve(fisher2);
    //  
    // M_->u_->col(0) += solu.segment(XtXW.rows(),LZWZL.rows());
    // beta_ += solu.segment(0,XtXW.rows());
    // 
    
    //Rcpp::Rcout << "\nbin: \n"<< bincr.transpose() << "\nvincr:\n" << vincr.transpose();
    
     
    M_->u_->col(0) += vincr;
    beta_ += bincr;
    sigma_ = sigmas;
  }
  
  //hessian of model parametere
  Eigen::MatrixXd b_hessian(double tol = 1e-4){
    glmmr::likelihood::L_likelihood hdl(M_,true,sigma_);
    std::vector<double> ndep;
    for(int i = 0; i < M_->P_; i++) ndep.push_back(tol);
    hdl.os.ndeps_ = ndep;
    std::vector<double> hessian(M_->P_ * M_->P_,0.0);
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    hdl.Hessian(start_b,hessian);
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),M_->P_,M_->P_);
    return hess;
  }
  

  Eigen::VectorXd f_grad(double tol = 1e-4){
    glmmr::likelihood::F_likelihood<T> fdl(D_,M_,cov_par_fix_,false,true,sigma_);
    fdl.os.usebounds_ = 1;
    std::vector<double> lower = lower_b_;
    std::vector<double> upper = upper_b_;
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    for(int i=0; i< theta_.size(); i++){
      lower.push_back(lower_t_[i]);
      upper.push_back(upper_t_[i]);
      start_b.push_back(theta_(i));
    } 
    fdl.os.lower_ = lower;
    fdl.os.upper_ = upper;
    std::vector<double> ndep;
    for(int i = 0; i < (M_->P_+Q_); i++) ndep.push_back(tol);
    fdl.os.ndeps_ = ndep;
    std::vector<double> gradient(M_->P_+Q_,0.0);
    fdl.Gradient(start_b,gradient);
    Eigen::VectorXd grad = Eigen::Map<Eigen::VectorXd>(gradient.data(),M_->P_+Q_);
    return grad;
  }
  
  Eigen::MatrixXd f_hess(double tol = 1e-4){
    glmmr::likelihood::F_likelihood<T> fhdl(D_,M_,cov_par_fix_,false,true,sigma_);
    fhdl.os.usebounds_ = 1;
    std::vector<double> lower = lower_b_;
    std::vector<double> upper = upper_b_;
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    for(int i=0; i< theta_.size(); i++){
      lower.push_back(lower_t_[i]);
      upper.push_back(upper_t_[i]);
      start_b.push_back(theta_(i));
    } 
    fhdl.os.lower_ = lower;
    fhdl.os.upper_ = upper;
    std::vector<double> ndep;
    for(int i = 0; i < (M_->P_+Q_); i++) ndep.push_back(tol);
    fhdl.os.ndeps_ = ndep;
    std::vector<double> hessian((M_->P_+Q_) * (M_->P_+Q_),0.0);

    fhdl.Hessian(start_b,hessian);
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),M_->P_+Q_,M_->P_+Q_);
    return hess;
  }
  
  Eigen::VectorXd get_beta(){
    return beta_;
  }
  
  Eigen::VectorXd get_theta(){
    return theta_;
  }
  
  double get_sigma(){
    return sigma_;
  }
  
};

}



#endif