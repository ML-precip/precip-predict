# Comparison Unets
library(ggplot2)
library(tidyverse)
library(viridis)

df <- read.csv("/Users/noeliaotero/Documents/CAS_ML/df_comparison_unet.csv")
df$X <- NULL

names(df) <- c("id", "n_params", "AUC", "Precision", "Recall", "N", "data")
df$data <- as.factor(df$data)
df$data <- factor(df$data, levels=c("train","test"))

met <- c("AUC","Precision", "Recall")
p_auc <- ggplot(df, aes(N, id, width = 5.9)) +
    geom_tile(aes(fill = AUC), colour = "grey90") + facet_grid(~data) + scale_x_continuous(breaks = df$N) +
    scale_fill_viridis(option="magma") +  ylab("")+ xlab("") +
    theme_bw()
p_prec <- ggplot(df, aes(N, id, width = 5.9)) +
  geom_tile(aes(fill = Precision), colour = "grey90") + facet_grid(~data) + scale_x_continuous(breaks = df$N)  +  
  scale_fill_viridis(option="magma") + ylab("")+ xlab("") +
  theme_bw()

p_rec <- ggplot(df, aes(N, id, width = 5.9)) +
  geom_tile(aes(fill = Recall), colour = "grey90") + facet_grid(~data) + scale_x_continuous(breaks = df$N) +
  scale_fill_viridis(option="magma") + ylab("")+ xlab("") +
  theme_bw()

fplot = ggpubr::ggarrange(p_auc, p_prec, p_rec, ncol=1)

ggsave(fplot, file="/Users/noeliaotero/Documents/CAS_ML/Fig_comparison_UNET.png", width = 10, height = 10)


##############
# Plot lines
#############

p1 <- df%>%ggplot2::ggplot(aes(N,AUC, color=id)) + geom_line()+ geom_point() + facet_grid(~data) +
  scale_x_continuous(breaks = df$N) +
  scale_color_brewer(palette="Set1",name="") +  ylab("")+ xlab("") + ggtitle("AUC") +
  theme_bw()+theme(text = element_text(size=15)) 


p2 <- df%>%ggplot2::ggplot(aes(N,Precision, color=id)) + geom_line() + geom_point() + facet_grid(~data) +
  scale_x_continuous(breaks = df$N) +
  scale_color_brewer(palette="Set1",name="") +  ylab("")+ xlab("") +ggtitle("Precision") +
  theme_bw() +
  theme(text = element_text(size=15)) 

p3 <- df%>%ggplot2::ggplot(aes(N,Recall, color=id)) + geom_line() + geom_point() + facet_grid(~data) +
  scale_x_continuous(breaks = df$N) +
  scale_color_brewer(palette="Set1", name="") +  ylab("")+ xlab("") + ggtitle("Recall") +
  theme_bw() + theme(text = element_text(size=15)) 

f2_plot = ggpubr::ggarrange(p1, p2, p3, ncol=1)
ggsave(f2_plot, file="/Users/noeliaotero/Documents/CAS_ML/Fig_comparison_UNET_lines.png", width = 10, height = 10)


###########################
# Plot ranked variables
##########################

df_rank <- read.csv("/Users/noeliaotero/Documents/CAS_ML/df_sort_list_crop_train.csv")

p <- ggplot(df_rank, aes(x = reorder(Variable, -Values), y = Values)) +
  geom_bar(stat = "identity", fill= "steelblue3", alpha=0.9) + ylab("") + 
  scale_y_continuous(breaks=NULL) + 
  xlab("")  +  theme_classic() + theme(text = element_text(size=20),
                     axis.text.x = element_text(angle=90, hjust=1)) 
 

ggsave(p, file="/Users/noeliaotero/Documents/CAS_ML/Fig_features_LRP.png", width = 16, height = 8)


# Plot number of parameters

# pt <- df%>%dplyr::filter(data=="train")%>%ggplot2::ggplot(aes(y=n_params, x=id)) + geom_bar(stat="identity") + 
#       scale_color_brewer(palette="Set1", name="") +  ylab("")+ xlab("") + ggtitle("Number of trainable parameters") +
#       theme_bw() + theme(text = element_text(size=12)) 

pt <- df%>%dplyr::filter(data=="train")%>%ggplot2::ggplot(aes(y=n_params, x=N, fill=id)) + geom_bar(stat="identity", position="dodge") + 
  scale_fill_brewer(palette="Set1", name="") +  ylab("")+ xlab("") + ggtitle("Number of trainable parameters") +
  theme_bw() + theme(text = element_text(size=20)) 

pt <- pt + scale_y_continuous(labels = scales::comma)
ggsave(pt, file="/Users/noeliaotero/Documents/CAS_ML/Num_param_unets.png", width = 16, height = 8)
