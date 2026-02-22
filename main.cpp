#include "solver.hpp"
#include <cassert>
#include <iostream>
int main(){
 Eigen::VectorXd x(8);x<<0,0,9,1,11,6,-1,4;
 std::vector<Constraint> c{fixed(0,0,8),fixed(1,0,8),horizontal(0,1,8),distance(0,1,10,8),distance(1,2,5,8),distance(2,3,10,8),distance(3,0,5,8),direction(0,1,1,2,true,8),direction(0,1,3,2,false,8)};
 auto r=solve(x,c);assert(r.error<1e-7);assert(r.rank==8);svg(r.points,"solution.svg");std::cout<<"residual="<<r.error<<" rank="<<r.rank<<" iterations="<<r.iterations<<"\n";
 auto under=solve(x,{distance(0,1,10,8)});assert(under.rank<8);
}
