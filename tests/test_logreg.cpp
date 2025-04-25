#include "nanoml/logistic_regression.hpp"
#include "nanoml/optim.hpp"
#include "nanoml/loss.hpp"
#include <iostream>

int main(){
    // OR gate
    ml::Matrix X(4,2);
    ml::Vector y(4);
    X(0,0)=0; X(0,1)=0; y[0]=0;
    X(1,0)=0; X(1,1)=1; y[1]=1;
    X(2,0)=1; X(2,1)=0; y[2]=1;
    X(3,0)=1; X(3,1)=1; y[3]=1;

    ml::LogisticRegression model(2);
    ml::GradientDescent gd({.lr=0.5, .epochs=2000, .verbose=false});
    gd.fit(model, X, y, ml::bce, ml::bce_grad);

    auto pred = model.forward(X);
    bool ok = pred[0]<0.5 && pred[1]>0.5 && pred[2]>0.5 && pred[3]>0.5;
    std::cout << "OR gate accuracy: " << ok << '\n';
    return ok?0:1;
}
