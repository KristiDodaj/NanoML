#include "nanoml/csv.hpp"
#include <iostream>
#include <fstream>
#include <cassert>

// Helper function to create a sample CSV file for testing
void create_sample_csv(const std::string& filename) {
    std::ofstream file(filename);
    file << "Name,Age,Height,Weight\n";
    file << "John,25,180.5,75.2\n";
    file << "Jane,30,165.3,55.8\n";
    file << "Mike,42,175.0,82.5\n";
    file << "Sarah,28,162.1,50.3\n";
}

int main() {
    const std::string test_filename = "test_data.csv";
    
    // Create a sample CSV file
    create_sample_csv(test_filename);
    
    // Test CSVReader constructor with filename
    ml::CSVReader reader(test_filename);
    
    std::cout << "Testing CSV Reader..." << std::endl;
    
    // Test if file loaded correctly
    assert(reader.is_loaded());
    std::cout << "File loaded successfully." << std::endl;
    
    // Test header reading
    assert(reader.headers().size() == 4);
    assert(reader.headers()[0] == "Name");
    assert(reader.headers()[1] == "Age");
    assert(reader.headers()[2] == "Height");
    assert(reader.headers()[3] == "Weight");
    std::cout << "Headers read correctly." << std::endl;
    
    // Test data reading
    assert(reader.rows() == 4);
    assert(reader.cols() == 4);
    std::cout << "Data dimensions correct: " << reader.rows() << " rows, " 
              << reader.cols() << " columns." << std::endl;
    
    // Test get_column by index
    ml::Vector ages = reader.get_column(1);
    assert(ages.size() == 4);
    assert(ages[0] == 25.0);
    assert(ages[1] == 30.0);
    assert(ages[2] == 42.0);
    assert(ages[3] == 28.0);
    std::cout << "Column by index (Age) read correctly." << std::endl;
    
    // Test get_column by name
    ml::Vector heights = reader.get_column("Height");
    assert(heights.size() == 4);
    assert(heights[0] == 180.5);
    assert(heights[1] == 165.3);
    assert(heights[2] == 175.0);
    assert(heights[3] == 162.1);
    std::cout << "Column by name (Height) read correctly." << std::endl;
    
    // Test get_columns by indices
    ml::Matrix age_height = reader.get_columns({1, 2});
    assert(age_height.rows() == 4);
    assert(age_height.cols() == 2);
    assert(age_height(0, 0) == 25.0);
    assert(age_height(0, 1) == 180.5);
    std::cout << "Multiple columns by indices read correctly." << std::endl;
    
    // Test get_columns by names
    ml::Matrix height_weight = reader.get_columns(std::vector<std::string>{"Height", "Weight"});
    assert(height_weight.rows() == 4);
    assert(height_weight.cols() == 2);
    assert(height_weight(2, 0) == 175.0);
    assert(height_weight(2, 1) == 82.5);
    std::cout << "Multiple columns by names read correctly." << std::endl;
    
    // Test to_matrix
    ml::Matrix all_data = reader.to_matrix();
    assert(all_data.rows() == 4);
    assert(all_data.cols() == 4);
    std::cout << "Full matrix conversion successful." << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    
    std::remove(test_filename.c_str());
    
    return 0;
}