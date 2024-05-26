library(dplyr)
library(readr)

# Load the data from CSV
data <- read_csv("testing/testResults/parameter search_2024_05_19_HSO/parameterSearchResultsHso.csv")

# Ensure that 'fitness' values are clean by removing square brackets and converting to numeric
data$fitness <- as.numeric(gsub("\\[|\\]", "", data$fitness))

# Calculate averages and retain other columns for each N and set combination
detailed_summary <- data %>%
  group_by(N, set) %>%
  summarise(
    avg_fitness = mean(fitness, na.rm = TRUE),  # Compute average fitness
    avg_particles = mean(particles, na.rm = TRUE),  # Compute average particles
    HM_size = first(`HM size`),  # Retain the first HM_size for each group, adjusted for space in column name
    hmcr = first(hmcr),  # Retain the first value of hmcr for each group
    par = first(par),  # Retain the first value of par for each group
    pb = first(pb),  # Retain the first value of pb for each group
    .groups = "drop"  # Drop the grouping structure after summarising
  )

# Save the data frame to a CSV file
write_csv(detailed_summary, "TestResults/average_per_set_HSO.csv")

# Provide a message to confirm the file has been saved
cat("The data has been saved to 'TestResults/average_per_set_HSO.csv'")
