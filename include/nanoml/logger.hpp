#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <map>
#include <sstream>

namespace ml {
    class Logger {
    public:
        struct MetricRecord {
            size_t epoch;
            double value;
        };

        enum class LogLevel {
            DEBUG,
            INFO,
            WARNING,
            ERROR
        };

        Logger(bool verbose = false, LogLevel level = LogLevel::INFO) 
            : verbose_(verbose), level_(level), start_time_(std::chrono::steady_clock::now()) {}

        void log(const std::string& message, LogLevel level = LogLevel::INFO) {
            if (!verbose_ || level < level_) return;
            
            std::string prefix;
            switch (level) {
                case LogLevel::DEBUG:   prefix = "[DEBUG] "; break;
                case LogLevel::INFO:    prefix = "[INFO] "; break;
                case LogLevel::WARNING: prefix = "[WARNING] "; break;
                case LogLevel::ERROR:   prefix = "[ERROR] "; break;
            }
            
            std::cout << prefix << message << std::endl;
        }

        void record_metric(const std::string& metric_name, double value, size_t epoch) {
            auto it = metrics_.find(metric_name);
            if (it == metrics_.end()) {
                metrics_[metric_name] = std::vector<MetricRecord>();
            }
            metrics_[metric_name].push_back({epoch, value});
        }

        void update_progress(size_t current, size_t total, 
                            const std::string& prefix = "", 
                            const std::string& suffix = "") {
            if (!verbose_) return;

            float percentage = static_cast<float>(current) / static_cast<float>(total);
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
            
            int estimated_total = (current > 0) ? 
                static_cast<int>(elapsed * total / current) : 0;
            int estimated_remaining = estimated_total - elapsed;

            std::string elapsed_str = format_time(elapsed);
            std::string remaining_str = format_time(estimated_remaining);
            
            int bar_width = 30;
            int pos = bar_width * percentage;
            
            std::cout << "\r" << prefix << " [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            
            std::cout << "] " << int(percentage * 100.0) << "% "
                    << suffix << " | " << elapsed_str << " < " << remaining_str;
            
            std::cout.flush();
            
            if (current == total) {
                std::cout << std::endl;
            }
        }

        const std::map<std::string, std::vector<MetricRecord>>& get_metrics() const {
            return metrics_;
        }

    private:
        bool verbose_;
        LogLevel level_;
        std::chrono::time_point<std::chrono::steady_clock> start_time_;
        std::map<std::string, std::vector<MetricRecord>> metrics_;
        
        std::string format_time(int seconds) const {
            int hours = seconds / 3600;
            int minutes = (seconds % 3600) / 60;
            int secs = seconds % 60;
            
            std::stringstream ss;
            if (hours > 0) {
                ss << hours << "h";
            }
            if (hours > 0 || minutes > 0) {
                ss << minutes << "m";
            }
            ss << secs << "s";
            return ss.str();
        }
    };
}