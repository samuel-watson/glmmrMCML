#ifndef GLMMRMCML_H
#define GLMMRMCML_H

#include <cmath>  
#include <RcppArmadillo.h>
#include <glmmr.h>

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
        0.5*((y(i) - mu(i))/var_par)*((y(i) - mu(i))/var_par);
    }
    break;
  case 8:
    logl = 0;
  }
  return logl;
}


#endif