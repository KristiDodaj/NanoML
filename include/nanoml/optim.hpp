#pragma once
#include "nanoml/model.hpp"
#include "nanoml/logger.hpp"
#include <iostream>
#include <memory>
#include <limits>

namespace ml {
    struct GDConfig {
        double       lr          = 0.01;
        std::size_t  epochs      = 1000;
        bool         verbose     = false;
        std::size_t  log_interval = 100;
        bool         show_progress_bar = true;
    };

class GradientDescent {
    public:
        explicit GradientDescent(GDConfig cfg = {}) 
            : cfg_(cfg), 
              logger_(std::make_shared<Logger>(cfg.verbose)) {}
        
        std::shared_ptr<Logger> get_logger() const {
            return logger_;
        }

        template<typename Loss, typename LossGrad>
        void fit(Model&       model,
                const Matrix& X,
                const Vector& y,
                Loss         loss_fn,
                LossGrad     grad_fn)
        {
            logger_->log("Starting training with " + std::to_string(cfg_.epochs) + " epochs", Logger::LogLevel::INFO);
            logger_->log("Learning rate: " + std::to_string(cfg_.lr), Logger::LogLevel::INFO);
            
            std::vector<double> losses;
            double best_loss = std::numeric_limits<double>::max();
            
            for (std::size_t e = 0; e < cfg_.epochs; ++e) {
                Vector y_hat = model.forward(X);
                double L = loss_fn(y_hat, y);

                logger_->record_metric("loss", L, e);
                losses.push_back(L);

                if (L < best_loss) {
                    best_loss = L;
                    logger_->log("New best loss: " + std::to_string(L) + " at epoch " + std::to_string(e), 
                                Logger::LogLevel::DEBUG);
                }

                model.backward(X, grad_fn(y_hat, y), cfg_.lr);

                if (cfg_.verbose && (e % cfg_.log_interval == 0 || e == cfg_.epochs-1)) {
                    std::string suffix = "loss = " + std::to_string(L);
                    logger_->log("epoch " + std::to_string(e) + ", " + suffix, Logger::LogLevel::INFO);
                }

                if (cfg_.show_progress_bar) {
                    std::string suffix = "loss = " + std::to_string(L);
                    logger_->update_progress(e+1, cfg_.epochs, "Training", suffix);
                }
            }
            
            logger_->log("Training completed. Final loss: " + std::to_string(losses.back()), 
                        Logger::LogLevel::INFO);
        }
    private:
        GDConfig cfg_;
        std::shared_ptr<Logger> logger_;
    };
}
