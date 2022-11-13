#ifndef MOREMATHS_H
#define MOREMATHS_H

#define _USE_MATH_DEFINES

#include <cmath> 
#include <unordered_map>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
  namespace maths {
  
  inline double log_factorial_approx(int n){
    double ans;
    if(n==0){
      ans = 0;
    } else {
      ans = n*log(n) - n + log(n*(1+4*n*(1+2*n)))/6 + log(M_PI)/2;
    }
    return ans;
  }
  
  inline double log_likelihood(double y,
                               double mu,
                               double var_par,
                               std::string family,
                               std::string link) {
    double logl;
    
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
      {
      double lf1 = glmmr::maths::log_factorial_approx(y);
      logl = y*mu - exp(mu)-lf1;
      break;
      }
    case 2:
      {
        double lf1 = log_factorial_approx(y);
        logl = y*log(mu) - mu-lf1;
        break;
      }
    case 3:
      if(y==1){
        logl = log(1/(1+exp(-mu)));
      } else if(y==0){
        logl = log(1 - 1/(1+exp(-mu)));
      }
      break;
    case 4:
      if(y==1){
        logl = mu;
      } else if(y==0){
        logl = log(1 - exp(mu));
      }
      break;
    case 5:
      logl = 0;
      break;
    case 6:
      logl = 0;
      break;
    case 7:
      logl = -1*log(var_par) -0.5*log(2*M_PI) -
        0.5*((y - mu)/var_par)*((y - mu)/var_par);
      break;
    case 8:
      logl = 0;
    }
    return logl;
  }
  
  // inline double log_likelihood(Eigen::VectorXd y,
  //                              Eigen::VectorXd mu,
  //                              double var_par,
  //                              std::string family,
  //                              std::string link) {
  //   double logl = 0;
  //   int n = y.size();
  //   int i;
  //   for(int i = 0; i<n; i++){
  //     logl += log_likelihood(y(i),mu(i),var_par,family,link);
  //   }
  //   return logl;
  // }
  
  
  }

namespace algo {
inline Eigen::VectorXd forward_sub(Eigen::MatrixXd* U,
                                   Eigen::VectorXd* u,
                                   int n)
{
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; i++) {
    double lsum = 0;
    for (int j = 0; j < i; j++) {
      lsum += (*U)(i,j) * y(j);
    }
    y(i) = ((*u)(i) - lsum) / (*U)(i,i);
  }
  return y;
}
}

}





#endif