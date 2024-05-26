library(dplyr)
library(readr)

# Load the data
data <- read_csv("TestResults/parameterSearchResultsPso.csv")

# Calculate averages and retain other columns for each N and set combination
detailed_summary <- data %>%
  group_by(N, set) %>%
  summarise(
    avg_fitness = mean(fitness, na.rm = TRUE),  # Compute average fitness
    avg_particles = mean(particles, na.rm = TRUE),  # Compute average particles
    population_size = first(population_size),  # Retain the first population_size for each group
    w = first(w),  # Retain the first value of w for each group
    c1 = first(c1),  # Retain the first value of c1 for each group
    c2 = first(c2),  # Retain the first value of c2 for each group
    .groups = "drop"  # Drop the grouping structure after summarising
  )

# Save the data frame to a CSV file
write_csv(average_per_set, "TestResults/average_per_set_PSO.csv")

# Provide a message to confirm the file has been saved
cat("The data has been saved to 'average_per_set.csv'")