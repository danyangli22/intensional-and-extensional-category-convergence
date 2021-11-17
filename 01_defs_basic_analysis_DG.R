#Color words 
rm(list=ls());gc()
source("C:/Users/dougl/Desktop/Categories/code/Functions/general/01_general_functions_DG.R")
library(plyr)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(tidyr)
library(clinfun)
library(wesanderson)
pal <- wes_palette("Zissou1", 100, type = "continuous")
min_max_norm<-function(x){(x - min(x,na.rm=T))/(max(x,na.rm=T) - min(x,na.rm=T))}

#Organize data
dt<-read.csv("C:/Users/dougl/Desktop/FULL_dis.csv")

dt_DG<-read.csv("C:/Users/dougl/Desktop/FULL_dis_DG.csv")
colnames(dt_DG)<-c("N","instance_id","label","cos_sim")

dt<-merge(dt, dt_DG, by=c("N","instance_id","label"))
hist(dt$cos_sim)

###
mod<-lm(success_rate ~ cos_sim * N, data=dt)
summary(mod)

dt_agg<- dt %>% group_by(N) %>% 
  dplyr::summarise(stat=cor.test(success_rate, cos_sim)$estimate, 
                   pval=cor.test(success_rate, cos_sim)$p.value, 
                   cilow=cor.test(success_rate, cos_sim)$conf.int[1],
                   cihi=cor.test(success_rate, cos_sim)$conf.int[2])

dt_agg

#get distance by property of the word 
kruskal.test(dt$cos_sim, dt$N)

ggplot(dt_agg, aes(x=as.factor(N), y = stat, ymin=cilow, ymax = cihi)) + 
  geom_point(size=6) + dougtheme_mod + 
  geom_errorbar(width = 0, size = 1) + 
  ylab("Correlation\n(Intensional ~ Extensional)") + xlab("N") + 
  theme(axis.text=element_text(size=30), 
        axis.title=element_text(size=30,  vjust = 0.5), 
        axis.title.x=element_text(size=30,  vjust = 0.5),
        axis.title.y=element_text(size=30,  vjust = 0.5)) + 
  geom_hline(yintercept = 0)

savepath<-"C:/Users/dougl/Desktop/Categories/Definitions_Ext/Results/"
ggsave('corr_ext_int.png', width=10, height=10, path = savepath)

###########################
load("C:/Users/dougl/Desktop/Categories/main/Data/Salience/all_labels_salience.Rdata"); head(all_raw_salience_agg)
dt_merge<-merge(dt, all_raw_salience_agg, by=c("label"))

cor.test(dt_merge$prop_intro, dt_merge$success_rate)
cor.test(dt_merge$prop_intro, dt_merge$cos_sim)
cor.test(dt_merge$z_salience, dt_merge$cos_sim)

dt_merge<-dt_merge %>% mutate(salience_bin=ntile(prop_intro,4))
dt_merge_agg<-dt_merge %>% group_by(salience_bin) %>% 
  dplyr::summarise(cos_sim = mean(cos_sim))
dt_merge_agg

mod2<-lm(cos_sim ~ prop_intro + N, data = dt_merge)
summary(mod2)

##################
#Just within N=24#
##################
n24<-subset(dt, N==24)
cor.test(n24$cos_sim, n24$success_rate)
mod<-lm(success_rate ~ distance + num_imgs + norm_image_breadth, data=n24)
summary(mod)

n24_label<-n24 %>% group_by(label) %>% 
  dplyr::summarise(distance=mean(distance, na.rm=T))

