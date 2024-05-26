# Install and load necessary packages
if (!require("ggplot2")) install.packages("ggplot2", dependencies=TRUE)
library(ggplot2)
library(dplyr)

# Assume the data is read from two CSV files stored in the working directory
hso_data <- read.csv("testing/testResults/testResultsHso.csv")
pso_data <- read.csv("testing/testResults/testResultsPso.csv")

# Ensuring fitness is numeric and adding method identifiers
hso_data$fitness <- as.numeric(gsub("\\[|\\]", "", hso_data$fitness))
pso_data$fitness <- as.numeric(gsub("\\[|\\]", "", pso_data$fitness))
hso_data$method <- "HSO"
pso_data$method <- "PSO"

# Combine the datasets, ensuring consistent columns are used
combined_data <- rbind(hso_data[, c("N", "fitness", "method", "id")], pso_data[, c("N", "fitness", "method", "id")])

# Normalize fitness values for each method so that the maximum is 100
max_fitness_hso <- max(combined_data$fitness[combined_data$method == "HSO"], na.rm = TRUE)
max_fitness_pso <- max(combined_data$fitness[combined_data$method == "PSO"], na.rm = TRUE)

combined_data$fitness <- ifelse(combined_data$method == "HSO",
                                (combined_data$fitness / max_fitness_hso) * 100,
                                (combined_data$fitness / max_fitness_pso) * 100)

# Filter data to only include N values from 2 to 5
filtered_data <- combined_data %>%
  filter(N %in% 2:5) %>%
  mutate(N = as.factor(N))  # Ensure N is treated as a factor

# Calculate ID group (block of 20 IDs)
filtered_data <- filtered_data %>%
  mutate(id_group = ceiling(id / 20))  # Group ids in blocks of 20

# Initialize a list to store plots
plot_list <- list()

# Generate a plot for each N value
for(n_value in levels(filtered_data$N)) {
  dodge <- position_dodge(width=0.9)  # Consistent placement and slight separation
  
  p <- ggplot(filtered_data[filtered_data$N == n_value,], aes(x=factor(id_group), y=fitness, fill=method)) +
    geom_boxplot(position=dodge, outlier.size=1.5) +
    labs(
         x="Image Set", y="Fitness %") +
    theme_minimal() +
    theme(text = element_text(size=15),  # General text size for the plot
          plot.title = element_text(size=17, face="bold"),  # Title size
          axis.title = element_text(size=15),  # Axis titles
          axis.text = element_text(size=13),  # Axis text (ticks)
          legend.title = element_text(size=15),  # Legend title
          legend.text = element_text(size=13)) +  # Legend items
    scale_fill_manual(values=c("HSO" = "aliceblue", "PSO" = "lightgreen"),
                      guide=guide_legend(title="Method", title.position="top", title.theme=element_text(size=14),
                                         label.theme=element_text(size=12)))  # Manual scale adjustment for legend
  
  plot_list[[n_value]] <- p
}

# Print and save plots
for(i in 1:length(plot_list)) {
  print(plot_list[[i]])
  ggsave(sprintf("testing/plots/average_fitness_comparison_N_%s.png", names(plot_list)[i]), plot=plot_list[[i]], width=10, height=8)
}

