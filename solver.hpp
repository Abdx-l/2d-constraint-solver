#pragma once
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <cmath>
#include <fstream>
struct Constraint{std::function<double(const Eigen::VectorXd&)> residual;std::function<Eigen::RowVectorXd(const Eigen::VectorXd&)> jacobian;};
struct Result{Eigen::VectorXd points;double error;int rank;int iterations;};
inline Constraint fixed(int coordinate,double value,int n){
 return {[=](auto&x){return x[coordinate]-value;},[=](auto&){Eigen::RowVectorXd j=Eigen::RowVectorXd::Zero(n);j[coordinate]=1;return j;}};
}
inline Constraint distance(int a,int b,double target,int n){
 return {[=](auto&x){return (x.template segment<2>(2*a)-x.template segment<2>(2*b)).squaredNorm()-target*target;},[=](auto&x){Eigen::RowVectorXd j=Eigen::RowVectorXd::Zero(n);Eigen::Vector2d d=2*(x.template segment<2>(2*a)-x.template segment<2>(2*b));j.template segment<2>(2*a)=d.transpose();j.template segment<2>(2*b)=-d.transpose();return j;}};
}
inline Constraint horizontal(int a,int b,int n){
 return {[=](auto&x){return x[2*a+1]-x[2*b+1];},[=](auto&){Eigen::RowVectorXd j=Eigen::RowVectorXd::Zero(n);j[2*a+1]=1;j[2*b+1]=-1;return j;}};
}
inline Constraint direction(int a,int b,int c,int d,bool perpendicular,int n){
 auto residual=[=](const Eigen::VectorXd&x){Eigen::Vector2d u=x.segment<2>(2*b)-x.segment<2>(2*a),v=x.segment<2>(2*d)-x.segment<2>(2*c);return perpendicular?u.dot(v):u.x()*v.y()-u.y()*v.x();};
 auto jac=[=](const Eigen::VectorXd&x){Eigen::RowVectorXd j=Eigen::RowVectorXd::Zero(n);Eigen::Vector2d u=x.segment<2>(2*b)-x.segment<2>(2*a),v=x.segment<2>(2*d)-x.segment<2>(2*c);Eigen::Vector2d du,dv;if(perpendicular){du=v;dv=u;}else{du<<v.y(),-v.x();dv<<-u.y(),u.x();}j.segment<2>(2*a)-=du.transpose();j.segment<2>(2*b)+=du.transpose();j.segment<2>(2*c)-=dv.transpose();j.segment<2>(2*d)+=dv.transpose();return j;};
 return {residual,jac};
}
inline Result solve(Eigen::VectorXd x,const std::vector<Constraint>&constraints,int limit=200){
 Eigen::VectorXd r(constraints.size());Eigen::MatrixXd j(constraints.size(),x.size());int iteration=0;
 for(;iteration<limit;iteration++){
  for(size_t i=0;i<constraints.size();i++){r[i]=constraints[i].residual(x);j.row(i)=constraints[i].jacobian(x);}
  if(r.norm()<1e-9)break;
  Eigen::VectorXd step=j.colPivHouseholderQr().solve(-r);
  double alpha=1;bool accepted=false;
  while(alpha>1e-8){Eigen::VectorXd trial=x+alpha*step;double error=0;for(auto&c:constraints)error+=std::pow(c.residual(trial),2);if(error<r.squaredNorm()){x=trial;accepted=true;break;}alpha*=.5;}
  if(!accepted)break;
 }
 for(size_t i=0;i<constraints.size();i++){r[i]=constraints[i].residual(x);j.row(i)=constraints[i].jacobian(x);}
 return {x,r.norm(),int(j.fullPivLu().rank()),iteration};
}
inline void svg(const Eigen::VectorXd&x,const char*file){
 std::ofstream f(file);f<<"<svg xmlns='http://www.w3.org/2000/svg' viewBox='-20 -20 200 200'><g fill='none' stroke='black'><polygon points='";
 for(int i=0;i<x.size()/2;i++)f<<x[2*i]*10<<","<<x[2*i+1]*10<<" ";
 f<<"'/></g></svg>";
}
