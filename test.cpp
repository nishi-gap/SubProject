#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <numbers>
#include <cmath>
#include <Eigen/Dense>
#include "nlopt.hpp"
const double eps = 1e-6;

typedef class{
public:
    Eigen::Vector2d e;
    double b;
}vec2d_b;

inline double fb(const std::vector<double>& X, std::vector<double> &grad, void *f_data){
    vec2d_b *eb = (vec2d_b*)f_data;
    double b = eb->b;
    Eigen::Vector3d e2 = Eigen::Vector3d(eb->e.x(),eb->e.y(),X[0]).normalized(); 
    double f = cos(b) - e2.dot(Eigen::Vector3d(1,0,0));
    e2 = Eigen::Vector3d(eb->e.x(),eb->e.y(),X[0] + eps).normalized(); 
    double fp = cos(b) - e2.dot(Eigen::Vector3d(1,0,0));
    e2 = Eigen::Vector3d(eb->e.x(),eb->e.y(),X[0] - eps).normalized(); 
    double fm = cos(b) - e2.dot(Eigen::Vector3d(1,0,0));
    grad[0] = (fp - fm)/(2.0*eps);
    return f;
}

Eigen::Vector3d getvec(double b, double k){
    Eigen::Matrix2d Rot; Rot << cos(k), -sin(k), sin(k), cos(k);
    Eigen::Vector2d init = (Rot * Eigen::Vector2d(1,0)); 
    vec2d_b eb(init, b);
    nlopt::opt opt;
    opt = nlopt::opt(nlopt::LD_MMA, 1);
    opt.set_min_objective(fb, &b);
    opt.set_xtol_rel(1e-13);
    opt.set_maxtime(5.0);//stop over this time
    opt.set_xtol_rel(1e-13);
    try {
        double fmin;
        std::vector<double> x{0};
        nlopt::result result = opt.optimize(x, fmin);
        return Eigen::Vector3d(init.x(), init.y(), x[0]).normalized();
    }catch (std::exception& e) {
          std::cout << "nlopt failed: " << e.what() <<std::endl;
          return Eigen::Vector3d(0,0,0);
    }
}

int main(){
    std::cout <<"hello"<<std::endl;
    double b = std::numbers::pi/2, k = std::numbers::pi/2;
    std::cout << getvec(b, k) << std::endl;
    return 0;
}