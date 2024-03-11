library(tidyverse)

# setwd("~/Projects/M4/examples/Tuolumne_figures/")

watershed = "Tuolumne_Watershed"
short_name = "Tuolumne"

watershed = "Blue_Dillon_Watershed"
short_name = "Blue-Dillon"

watershed = "Conejos_Watershed"
short_name = "Conejos"

watershed = "Dolores_Watershed"
short_name = "Dolores"



m4_example_dir = "~/Projects/M4/examples/"
setwd(m4_example_dir)



proc_model_data = function(folder_loc){
    
  files = list.files(folder_loc, pattern = "*AdditionalDataForPostProcessing.csv")
  
  model_dat = data.frame()
  for (file in files){
    print(file)
    df = read_csv(paste0(folder_loc, file))
    folder = strsplit(folder_loc, "/")[[1]]
    model_name = strsplit(file, "_")[[1]][1]
    
    df = df[, c(2, 7, 3, 5, 4, 6)]
    names(df) = c("year", "true", "train_pred", "test_pred", "train_res", "test_res")
    df$model = model_name
    df$folder = folder
    model_dat = rbind(model_dat, df)
  }
  
  pdat = gather(model_dat, -year, -model, -folder, key = type, value=value)
  return(pdat)

}


proc_accuracy = function(folder_loc){
    
  files = list.files(folder_loc, pattern = "*AdditionalReportingSuite.csv")
  
  acc_dat = data.frame()
  for (file in files){
    print(file)
    df = read_csv(paste0(folder_loc, file))
    folder = strsplit(folder_loc, "/")[[1]]
    model_name = strsplit(file, "_")[[1]][1]
    df = df[, c(2, 3, 4)]
    names(df) = c("metric", "insample", "outsample")
    df$model = model_name
    df$folder = folder
    acc_dat = rbind(acc_dat, df)
  }
  
  pdat = gather(acc_dat, -metric, -model, -folder, key = type, value=value)
  return(pdat)
  #
}


get_model_data = function(watershed, short_name){
  mod_bdat = proc_model_data(paste0(watershed, "_baseline/"))
  mod_adat = proc_model_data(paste0(watershed, "_aso_swe/"))
  mod_badat = proc_model_data(paste0(watershed, "_baseline_aso_swe/"))
  mod_batdat = proc_model_data(paste0(watershed, "_baseline_aso_swe_temp_precip/"))
  mod_batdat2 = proc_model_data(paste0(watershed, "_aso_temp_precip/"))
  
  pdat1 = rbind(mod_bdat, mod_adat)
  pdat1 = rbind(mod_bdat, mod_adat, mod_badat, mod_batdat, mod_batdat2)
  
  pdat1$folder = factor(pdat1$folder,                      
                        levels = c(paste0(watershed, "_aso_swe"), 
                                   paste0(watershed, "_baseline"), 
                                   paste0(watershed, "_baseline_aso_swe"), 
                                   paste0(watershed, "_baseline_aso_swe_temp_precip"),
                                   paste0(watershed, "_aso_temp_precip")),
                        labels = c(paste0(short_name, "-ASO"), 
                                   paste0(short_name, "-Baseline"), 
                                   paste0(short_name, "-Baseline + ASO"), 
                                   paste0(short_name, "-Baseline + ASO + Temp + Precip"),
                                   paste0(short_name, "-ASO + Temp + Precip")))
  
  
  return(pdat1)
}


get_acc_data = function(watershed, short_name){
  acc_bdat = proc_accuracy(paste0(watershed, "_baseline/"))
  acc_adat = proc_accuracy(paste0(watershed, "_aso_swe/"))
  acc_badat = proc_accuracy(paste0(watershed, "_baseline_aso_swe/"))
  acc_batdat = proc_accuracy(paste0(watershed, "_baseline_aso_swe_temp_precip/"))
  acc_batdat2 = proc_accuracy(paste0(watershed, "_aso_temp_precip/"))
  
  pdat2 = rbind(acc_bdat, acc_adat)
  pdat2 = rbind(acc_bdat, acc_adat, acc_badat, acc_batdat, acc_batdat2)
  pdat2
  
  pdat2$folder = factor(pdat2$folder, 
                        levels = c(paste0(watershed, "_aso_swe"), 
                                   paste0(watershed, "_baseline"), 
                                   paste0(watershed, "_baseline_aso_swe"), 
                                   paste0(watershed, "_baseline_aso_swe_temp_precip"),
                                   paste0(watershed, "_aso_temp_precip")),
                        labels = c(paste0(short_name, "-ASO"), 
                                   paste0(short_name, "-Baseline"), 
                                   paste0(short_name, "-Baseline + ASO"), 
                                   paste0(short_name, "-Baseline + ASO + Temp + Precip"),
                                   paste0(short_name, "-ASO + Temp + Precip")))
  
  return(pdat2)
}


pdat1 = get_model_data(watershed, short_name)
pdat2 = get_acc_data(watershed, short_name)


ggplot(filter(pdat2, metric %in% c("Rsqrd", "RMSE") & folder == paste0(short_name, "-Baseline")), aes(model, value, color=factor(type))) + 
  geom_point(shape=15, size=5) + 
  # ylim(0.75, 1) +
  theme_minimal(15) +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position = c(0.055, 0.89)) +
  facet_wrap(folder~metric, scales='free_y')



ggplot(filter(pdat2, metric %in% c("Rsqrd", "RMSE") & 
                folder %in% c(paste0(short_name, "-Baseline"), 
                              paste0(short_name, "-ASO")) & type == "outsample"), 
       aes(model, value, color=factor(folder))) + 
  geom_point(shape=15, size=5) + 
  # ylim(0.75, 1) +
  theme_minimal(15) +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position = "bottom") +
  facet_wrap(~metric, scales='free_y')



ggplot(filter(pdat1, type == "test_res" & folder %in% c(paste0(short_name, "-Baseline"), paste0(short_name, "-ASO"))), 
       aes(year, value, color=factor(folder))) + 
  geom_line() + 
  theme_minimal(15) + 
  labs(x=NULL, y="Test Residuals", color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1),
        legend.position = "bottom") +
  facet_wrap(~model) +
  NULL



ggplot(filter(pdat1, type == "test_res"), aes(year, value, color=factor(folder))) + 
  geom_line() + 
  theme_minimal() + 
  labs(x=NULL, y="Test Residuals", color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1),
        legend.position = "bottom") +
  facet_wrap(~model) +
  NULL


ggplot(filter(pdat2, metric %in% c("Rsqrd", "RMSE") & type == "outsample"), aes(model, value, color=factor(folder))) + 
  geom_point(shape=15, size=2) + 
  # ylim(0.75, 1) +
  theme_minimal() +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position = "bottom") +
  facet_wrap(~metric, scales='free_y')


ggplot(filter(pdat2, metric %in% c("Rsqrd") & type == "outsample"), aes(model, value, color=factor(folder))) + 
  geom_point(shape=15, size=5) + 
  # ylim(0.8, 1) +
  theme_minimal(15) +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position = "bottom") +
  facet_wrap(~metric, scales='free_y')


ggplot(filter(pdat2, metric %in% c("RMSE") & type == "outsample"), aes(model, value, color=factor(folder))) + 
  geom_point(shape=15, size=5) + 
  # ylim(0.8, 1) +
  theme_minimal(15) +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position = "bottom") +
  facet_wrap(~metric, scales='free_y')



# ------------------------------------
# All Watersheds
tuolumne_mdat = get_acc_data("Tuolumne_Watershed", "Tuolumne")
blueriver_mdat = get_acc_data("Blue_Dillon_Watershed", "Blue River")
dolores_mdat = get_acc_data("Dolores_Watershed", "Dolores")
conejos_mdat = get_acc_data("Conejos_Watershed", "Conejos")


watersheds = rbind(tuolumne_mdat, blueriver_mdat, dolores_mdat, conejos_mdat)

watersheds_rsq = filter(watersheds, type == "outsample" & model == "ensemble" & metric == "Rsqrd")
watersheds_rsq = watersheds_rsq %>%
                separate(col = folder, into = c("watershed", "model"), sep = "-", extra = "merge")

watersheds_rmse = filter(watersheds, type == "outsample" & model == "ensemble" & metric == "RMSE")
watersheds_rmse = watersheds_rmse %>%
                separate(col = folder, into = c("watershed", "model"), sep = "-", extra = "merge")

filter(watersheds_rmse, watershed == "Conejos")
filter(watersheds_rsq, watershed == "Conejos")
filter()

ggplot(watersheds_rsq, aes(watershed, value, color=model)) + 
  geom_point(shape=15, size=5) + 
  geom_label(aes(label = round(value, 2)), nudge_x = 0.1) +
  theme_minimal(15) +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position="bottom") +
  # facet_wrap(~model)
  NULL



watersheds_rsq = filter(watersheds, type == "outsample" & metric == "Rsqrd")

watersheds_rsq = watersheds_rsq %>%
                separate(col = folder, into = c("watershed", "new_model"), sep = "-", extra = "merge")


watersheds_rsq = watersheds_rsq %>% group_by(model) %>% 
                separate(col = folder, into = c("watershed", "model"), sep = "-", extra = "merge")



watersheds_rmse = filter(watersheds, type == "outsample" & metric == "RMSE")

watersheds_rmse = watersheds_rmse %>%
                separate(col = folder, into = c("watershed", "new_model"), sep = "-", extra = "merge")


watersheds_rmse = watersheds_rmse %>% group_by(model) %>% 
                separate(col = folder, into = c("watershed", "model"), sep = "-", extra = "merge")


tdat = filter(watersheds_rsq, watershed == "Dolores")
tdat = distinct(tdat)
View(tdat)

tdat = filter(watersheds_rmse, watershed == "Conejos")
tdat = distinct(tdat)
View(tdat)


ggplot(watersheds_rsq, aes(model, value, shape=new_model)) + 
  # geom_point(shape=15, size=5) +
  geom_point(size=4) + 
  theme_minimal(15) +
  # scale_color_viridis_d() +
  labs(x=NULL, y=NULL, color=NULL) +
  theme(panel.border = element_rect(color = "black", fill=NA, size=1), 
        legend.background = element_rect(color = "black", fill = NA, size = .5),
        legend.position="bottom") +
  facet_wrap(~watershed)
  NULL




