#pragma once
#include "nanoml/logger.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <functional>

namespace ml {
    /**
    * @brief Visualization class for plotting training metrics
    * 
    * This class provides functionality to visualize training metrics by generating
    * ASCII plots in the console or exporting data to CSV for external plotting.
    */
    class Visualizer {
    public:
        Visualizer() {}

        void plot_metrics_ascii(const Logger& logger, const std::string& metric_name) {
            const auto& metrics = logger.get_metrics();
            auto it = metrics.find(metric_name);
            
            if (it == metrics.end() || it->second.empty()) {
                std::cout << "No data available for metric: " << metric_name << std::endl;
                return;
            }
            
            const auto& records = it->second;

            double min_value = records[0].value;
            double max_value = records[0].value;
            size_t max_epoch = records[records.size() - 1].epoch;
            
            for (const auto& record : records) {
                min_value = std::min(min_value, record.value);
                max_value = std::max(max_value, record.value);
            }

            double range = max_value - min_value;
            min_value -= range * 0.05;
            max_value += range * 0.05;
            range = max_value - min_value;

            const int plot_height = 20;
            const int plot_width = 80;
            
            std::vector<std::vector<char>> plot(plot_height, std::vector<char>(plot_width, ' '));

            for (int i = 0; i < plot_height; ++i) {
                plot[i][0] = '|';
            }
            for (int i = 0; i < plot_width; ++i) {
                plot[plot_height - 1][i] = '-';
            }

            for (size_t i = 0; i < records.size(); ++i) {
                const auto& record = records[i];
                int x = static_cast<int>((record.epoch * (plot_width - 2)) / max_epoch) + 1;
                int y = plot_height - 1 - static_cast<int>((record.value - min_value) * (plot_height - 2) / range) - 1;
                
                if (x >= 0 && x < plot_width && y >= 0 && y < plot_height) {
                    plot[y][x] = '*';
                }
            }

            std::cout << "\nPlot of " << metric_name << ":" << std::endl;
            std::cout << std::fixed << std::setprecision(4);

            for (int i = 0; i < plot_height; ++i) {
                double y_value = max_value - (i * range / (plot_height - 2));
                if (i == 0 || i == plot_height - 1 || i == plot_height / 2) {
                    std::cout << std::setw(10) << y_value << " ";
                } else {
                    std::cout << std::setw(11) << " ";
                }

                for (int j = 0; j < plot_width; ++j) {
                    std::cout << plot[i][j];
                }
                std::cout << std::endl;
            }

            std::cout << std::setw(11) << " "; 
            std::cout << "0";
            std::cout << std::setw(plot_width - 2) << max_epoch;
            std::cout << std::endl;
        }

        bool export_metrics_csv(const Logger& logger, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }
            
            const auto& metrics = logger.get_metrics();

            file << "epoch";
            for (const auto& metric : metrics) {
                file << "," << metric.first;
            }
            file << std::endl;
            
            size_t max_epoch = 0;
            for (const auto& metric : metrics) {
                if (!metric.second.empty()) {
                    max_epoch = std::max(max_epoch, metric.second.back().epoch);
                }
            }

            std::map<std::string, std::map<size_t, double>> epoch_values;
            for (const auto& metric : metrics) {
                for (const auto& record : metric.second) {
                    epoch_values[metric.first][record.epoch] = record.value;
                }
            }

            for (size_t epoch = 0; epoch <= max_epoch; ++epoch) {
                file << epoch;
                
                for (const auto& metric : metrics) {
                    file << ",";
                    auto it = epoch_values[metric.first].find(epoch);
                    if (it != epoch_values[metric.first].end()) {
                        file << it->second;
                    }
                }
                
                file << std::endl;
            }
            
            return true;
        }

        bool export_metrics_html(const Logger& logger, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }
            
            const auto& metrics = logger.get_metrics();

            file << "<!DOCTYPE html>\n"
                << "<html>\n"
                << "<head>\n"
                << "  <title>NanoML Training Metrics</title>\n"
                << "  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n"
                << "</head>\n"
                << "<body>\n"
                << "  <div style=\"width:800px;margin:0 auto;\">\n"
                << "    <canvas id=\"metricsChart\"></canvas>\n"
                << "  </div>\n"
                << "  <script>\n"
                << "    const ctx = document.getElementById('metricsChart');\n"
                << "    const data = {\n"
                << "      labels: [";

            size_t max_epoch = 0;
            for (const auto& metric : metrics) {
                if (!metric.second.empty()) {
                    max_epoch = std::max(max_epoch, metric.second.back().epoch);
                }
            }

            for (size_t epoch = 0; epoch <= max_epoch; ++epoch) {
                file << epoch;
                if (epoch < max_epoch) {
                    file << ", ";
                }
            }
            
            file << "],\n";
            file << "      datasets: [\n";

            bool first_metric = true;
            for (const auto& metric : metrics) {
                if (!first_metric) {
                    file << ",\n";
                }
                first_metric = false;

                int r = (std::hash<std::string>{}(metric.first) % 256);
                int g = (std::hash<std::string>{}(metric.first + "g") % 256);
                int b = (std::hash<std::string>{}(metric.first + "b") % 256);
                
                file << "        {\n"
                    << "          label: '" << metric.first << "',\n"
                    << "          data: [";

                std::map<size_t, double> epoch_values;
                for (const auto& record : metric.second) {
                    epoch_values[record.epoch] = record.value;
                }

                for (size_t epoch = 0; epoch <= max_epoch; ++epoch) {
                    auto it = epoch_values.find(epoch);
                    if (it != epoch_values.end()) {
                        file << it->second;
                    } else {
                        file << "null";
                    }
                    
                    if (epoch < max_epoch) {
                        file << ", ";
                    }
                }
                
                file << "],\n"
                    << "          borderColor: 'rgb(" << r << ", " << g << ", " << b << ")',\n"
                    << "          tension: 0.1\n"
                    << "        }";
            }
            
            file << "\n      ]\n"
                << "    };\n"
                << "    new Chart(ctx, {\n"
                << "      type: 'line',\n"
                << "      data: data,\n"
                << "      options: {\n"
                << "        responsive: true,\n"
                << "        plugins: {\n"
                << "          title: {\n"
                << "            display: true,\n"
                << "            text: 'NanoML Training Metrics'\n"
                << "          },\n"
                << "        },\n"
                << "        interaction: {\n"
                << "          intersect: false,\n"
                << "        },\n"
                << "      }\n"
                << "    });\n"
                << "  </script>\n"
                << "</body>\n"
                << "</html>";
            
            return true;
        }
    };
}