library(bnlearn)

csv_filename <- "~/Documents/dev/bpm-causal/new_aggregate.csv";

# read data
data <- read.csv(csv_filename);
data[] <- lapply(data, as.factor)
# dat[] <- lapply(dat, as.numeric)

# define blacklisted edges
partial_order = list(
  c("Start.Event"), 
  c("Task.1", "Resource.1"), 
  c("Task.2", "Resource.2"), 
  c("Task.3"), c("End.Event"))
blacklist_edges <- tiers2blacklist(partial_order)

# add additional blacklisted edges
blacklist_edges <- rbind(blacklist_edges, r("Resource.1", "Resource.2"))
blacklist_edges <- rbind(blacklist_edges, c("Start.Event", "Resource.2"))

causal_model <- hc(data, blacklist = blacklist_edges)

plot(causal_model)