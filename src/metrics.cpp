#include "nanoml/metrics.hpp"
#include <cassert>

namespace ml {
    double mean_squared_error(const Vector& y_hat, const Vector& y) {
        assert(y_hat.size() == y.size());
        double sum = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double diff = y_hat[i] - y[i];
            sum += diff * diff;
        }
        return sum / y.size();
    }

    double r_squared(const Vector& y_hat, const Vector& y) {
        assert(y_hat.size() == y.size());
        double mean_y = 0.0;
        for (size_t i = 0; i < y.size(); ++i) mean_y += y[i];
        mean_y /= y.size();
        double ss_tot = 0.0, ss_res = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double diff = y[i] - mean_y;
            ss_tot += diff * diff;
            double res = y[i] - y_hat[i];
            ss_res += res * res;
        }
        if (ss_tot == 0.0) return 0.0;
        return 1.0 - ss_res / ss_tot;
    }

    std::map<std::string, int> confusion_matrix(const Vector& y_hat, const Vector& y) {
        assert(y_hat.size() == y.size());
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            bool pred = y_hat[i] > 0.5;
            bool actual = y[i] > 0.5;
            if (pred && actual) tp++;
            else if (pred && !actual) fp++;
            else if (!pred && !actual) tn++;
            else if (!pred && actual) fn++;
        }
        return {{"tp", tp}, {"fp", fp}, {"tn", tn}, {"fn", fn}};
    }

    double accuracy(const Vector& y_hat, const Vector& y) {
        auto cm = confusion_matrix(y_hat, y);
        int correct = cm["tp"] + cm["tn"];
        return y.size() ? static_cast<double>(correct) / y.size() : 0.0;
    }

    double precision(const Vector& y_hat, const Vector& y) {
        auto cm = confusion_matrix(y_hat, y);
        int tp = cm["tp"], fp = cm["fp"];
        int denom = tp + fp;
        return denom ? static_cast<double>(tp) / denom : 0.0;
    }

    double recall(const Vector& y_hat, const Vector& y) {
        auto cm = confusion_matrix(y_hat, y);
        int tp = cm["tp"], fn = cm["fn"];
        int denom = tp + fn;
        return denom ? static_cast<double>(tp) / denom : 0.0;
    }

    double f1_score(const Vector& y_hat, const Vector& y) {
        double p = precision(y_hat, y);
        double r = recall(y_hat, y);
        return (p + r) ? 2.0 * p * r / (p + r) : 0.0;
    }
}