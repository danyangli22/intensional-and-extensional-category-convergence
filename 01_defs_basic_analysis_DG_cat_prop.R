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
library(multiwayvcov)
library(lmtest)
pal <- wes_palette("Zissou1", 100, type = "continuous")
min_max_norm<-function(x){(x - min(x,na.rm=T))/(max(x,na.rm=T) - min(x,na.rm=T))}

#Organize data
dt<-read.csv("C:/Users/dougl/Desktop/FULL_dis.csv")

dt_DG<-read.csv("C:/Users/dougl/Desktop/FULL_dis_DG.csv")
colnames(dt_DG)<-c("N","instance_id","label","cos_sim")

dt<-merge(dt, dt_DG, by=c("N","instance_id","label"))
hist(dt$cos_sim)

#################

conc<-read.csv("C:/Users/dougl/Desktop/Categories/external data/Concreteness_ratings_Brysbaert_et_al_BRM.csv")
colnames(conc)[1]<-"label"

dt_conc<-merge(dt, conc, by=c("label"))

cor.test(dt_conc$Conc.M, dt_conc$cos_sim)
cor.test(dt_conc$Conc.M, dt_conc$N, method="spearman")

mod<-lm(cos_sim ~ Conc.M + N, data=dt_conc)
summary(mod)

dt_conc_vcov <- cluster.vcov(mod, dt_conc$instance_id)
coeftest(mod, dt_conc_vcov)

dt_conc_agg<-dt_conc %>% group_by(N) %>% 
  dplyr::summarise(
    concreteness=mean(Conc.M), 
    cilow=t.test(Conc.M)$conf.int[1], 
    cihi=t.test(Conc.M)$conf.int[2]
  )

ggplot(dt_conc_agg, aes(x=as.factor(N), y = concreteness, ymin=cilow, ymax = cihi)) + 
  geom_point(size=6) + dougtheme_mod + 
  geom_errorbar(width = 0, size = 1) + 
  ylab("Average Concreteness of \n Category") + xlab("N") + 
  theme(axis.text=element_text(size=30), 
        axis.title=element_text(size=30,  vjust = 0.5), 
        axis.title.x=element_text(size=30,  vjust = 0.5),
        axis.title.y=element_text(size=30,  vjust = 0.5)) 

savepath<-"C:/Users/dougl/Desktop/Categories/Definitions_Ext/Results/"
ggsave('concreteness.png', width=10, height=10, path = savepath)



################
amb<-read.csv("C:/Users/dougl/Desktop/Categories/external data/ambiguity_data.csv")
colnames(amb)[1]<-"label"

dt_amb<-merge(dt, amb, by=c("label"))

cor.test(dt_amb$SemD, dt_amb$N)

cor.test(dt_amb$SemD, dt_amb$cos_sim)
cor.test(dt_amb$mean_cos, dt_amb$cos_sim)
cor.test(dt_amb$SemD, dt_amb$success_rate)
cor.test(dt_amb$BNC_contexts, dt_amb$cos_sim)
cor.test(dt_amb$BNC_wordcount, dt_amb$cos_sim)

mod<-lm(SemD ~ N + success_rate + num_imgs, data=dt_amb)
summary(mod)

dt_amb_vcov <- cluster.vcov(mod, dt_amb$instance_id)
coeftest(mod, dt_amb_vcov)




