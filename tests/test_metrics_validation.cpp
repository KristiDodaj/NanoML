#include "nanoml/metrics.hpp"
#include "nanoml/validation.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/vector.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

// Helper for approximate comparison of doubles
bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

int main() {
    using ml::Vector;
    using ml::Matrix;
    using ml::mean_squared_error;
    using ml::r_squared;
    using ml::accuracy;
    using ml::precision;
    using ml::recall;
    using ml::f1_score;
    using ml::confusion_matrix;
    using ml::train_test_split;
    using ml::MinibatchGenerator;

    // Test regression metrics
    {
        Vector y{1.0, 2.0, 3.0};
        Vector yhat{1.0, 2.0, 4.0};
        double mse = mean_squared_error(yhat, y);
        assert(approx_equal(mse, (0.0+0.0+1.0)/3.0));
        double r2 = r_squared(yhat, y);
        // mean(y)=2, ss_tot=2, ss_res=1 => r2=0.5
        assert(approx_equal(r2, 0.5));
    }

    // Test classification metrics
    {
        Vector y{0,1,1,0};
        Vector yhat1{0.2, 0.8, 0.6, 0.4}; // preds: 0,1,1,0 => all correct
        assert(approx_equal(accuracy(yhat1,y), 1.0));
        assert(approx_equal(precision(yhat1,y), 1.0));
        assert(approx_equal(recall(yhat1,y), 1.0));
        assert(approx_equal(f1_score(yhat1,y), 1.0));
        auto cm = confusion_matrix(yhat1,y);
        assert(cm["tp"] == 2 && cm["tn"] == 2 && cm["fp"] == 0 && cm["fn"] == 0);
        Vector yhat2{0.6,0.4,0.4,0.6}; // preds:1,0,0,1 => all wrong
        assert(approx_equal(accuracy(yhat2,y), 0.0));
    }

    // Test train_test_split
    {
        size_t n = 5;
        Matrix X(n,1);
        Vector y(n);
        for (size_t i=0;i<n;++i) { X(i,0) = static_cast<double>(i); y[i] = static_cast<double>(i); }
        Matrix Xtr, Xval;
        Vector ytr, yval;
        train_test_split(X,y,Xtr,ytr,Xval,yval, 0.4);
        assert(Xtr.rows() == 3 && ytr.size() == 3);
        assert(Xval.rows() == 2 && yval.size() == 2);
        // ensure no overlap: combined sizes
        assert(Xtr.rows()+Xval.rows() == n);
    }

    // Test MinibatchGenerator
    {
        size_t n = 5;
        Matrix X(n,1);
        Vector y(n);
        double sum_all = 0.0;
        for (size_t i=0;i<n;++i) { X(i,0)=static_cast<double>(i+1); y[i]=static_cast<double>(i+1); sum_all += i+1; }
        MinibatchGenerator gen(X,y,2);
        double sum_batches = 0.0;
        size_t cnt_batches = 0;
        gen.reset();
        while (gen.has_next_batch()) {
            auto [Xb,yb] = gen.next_batch();
            for (size_t i = 0; i < yb.size(); ++i) sum_batches += yb[i];
            cnt_batches++;
        }
        // 5 items, batch_size=2 => 3 batches
        assert(cnt_batches == 3);
        assert(approx_equal(sum_batches, sum_all));
    }

    std::cout << "All metrics and validation tests passed ✅\n";
    return 0;
}
