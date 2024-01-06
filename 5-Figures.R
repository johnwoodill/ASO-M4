library(arrow)
library(tidyverse)
library(sf)

setwd("~/Projects/ASO-M4/")

waterbasin = "Tuolumne_Watershed"
shapefile_loc = "data/Tuolumne_Watershed/shapefiles/Tuolumne_Watershed.shp"

waterbasin = "Blue_Dillon_Watershed"
shapefile_loc = "blue_dillon_watershed/Blue_Dillon.shp"

waterbasin = "Dolores_Watershed"
shapefile_loc = "data/Dolores_Watershed/shapefiles/Dolores_waterbasin.shp"

waterbasin = "Conejos_Watershed"
shapefile_loc = "data/Conejos_Watershed/shapefiles/Conejos_waterbasin.shp"

adat = read_csv(paste0("~/Projects/ASO-M4/data/", waterbasin, "/processed/aso_basin_data.csv"))
edat = read_csv(paste0("~/Projects/ASO-M4/data/", waterbasin, "/processed/aso_elev_grade_aspect.csv"))
mdat = read_parquet(paste0("~/Projects/ASO-M4/data/", waterbasin, "/processed/model_data_elevation_prism_sinceSep_nlcd.parquet"))

sdat = read_sf(paste0(shapefile_loc))
sdat = st_transform(sdat, "EPSG:4326")

date_ = sample(adat$date, 1)


ggplot(filter(adat, date == date_), aes(lon, lat, color=SWE)) + 
  theme_minimal(15) +
  scale_fill_viridis_c() +
  # geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  # geom_tile() +
  geom_point() +
  labs(x=NULL, y=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1), 
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  NULL



# Elevation
ggplot(edat, aes(lon, lat, color=elev_m)) + 
  theme_minimal(15) +
  scale_fill_viridis_c() +
  # geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  # geom_tile() +
  geom_point() +
  labs(x=NULL, y=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1), 
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  NULL


# Aspect
ggplot(edat, aes(lon, lat, color=aspect)) + 
  theme_minimal(15) +
  scale_fill_viridis_c() +
  # geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  # geom_tile() +
  geom_point() +
  labs(x=NULL, y=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1), 
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  NULL



# Grade
ggplot(edat, aes(lon, lat, color=grade)) + 
  theme_minimal(15) +
  scale_fill_viridis_c() +
  # geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  # geom_tile() +
  geom_point() +
  labs(x=NULL, y=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1), 
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  NULL




prdat = read_csv(paste0("~/Projects/ASO-M4/data/", waterbasin, "/processed/", waterbasin, "_PRISM_daily_1981-2023.csv"))

prdat1 = filter(prdat, date == "20210215" & var == "ppt")

ggplot(prdat1, aes(longitude, latitude, fill=value)) + 
  geom_tile() +
  geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  scale_fill_viridis_c(limits = c(0, 4), na.value = "white") +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1),
    legend.position='bottom',
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  labs(x=NULL, y=NULL, fill=NULL) +
    guides(fill = guide_colorbar(title.position = "bottom", 
                             direction = "horizontal",
                             frame.colour = "black",
                             barwidth = 20,
                             barheight = 1)) +
  NULL
  

trdat = filter(prdat, date == "20210101" & var %in% c("tmin", "tmax"))
trdat = spread(trdat, key = var, value = value)
trdat$tmean = (trdat$tmax + trdat$tmin) / 2

ggplot(trdat, aes(longitude, latitude, fill=tmean)) + 
  geom_tile() +
  geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  scale_fill_viridis_c(limits = c(-10, 10), na.value = "white", option="B") +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1),
    legend.position='bottom',
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  labs(x=NULL, y=NULL, fill=NULL) +
    guides(fill = guide_colorbar(title.position = "bottom", 
                             direction = "horizontal",
                             frame.colour = "black",
                             barwidth = 20,
                             barheight = 1)) +
  NULL



trdat = filter(prdat, var %in% c("tmin", "tmax", "ppt"))
trdat = spread(trdat, key = var, value = value)
trdat$tmean = (trdat$tmax + trdat$tmin) / 2

trdat = filter(trdat, tmean <= 0 & ppt > 0)
trdat$year = substr(trdat$date, 1, 4)

trdat$lat_lon = paste0(trdat$latitude, "_", trdat$longitude)

trdat1 = trdat %>% 
  group_by(year, lat_lon) %>% 
  summarise(count_neg = n(),
            lat = mean(latitude),
            lon = mean(longitude)) %>% 
  ungroup() %>% 
  group_by(lat_lon) %>% 
  summarise(count_neg = mean(count_neg), 
            lat = mean(lat), 
            lon = mean(lon))

# ggplot(trdat1, aes(lon, lat, fill=count_neg)) + 
#   geom_tile() +
#   facet_wrap(~year)


trdat1 = trdat1 %>% dplyr::select(lat, lon) %>% distinct()

ggplot(trdat1, aes(lon, lat, fill=count_neg)) + 
  # geom_tile() 
  geom_point() +
  geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  # scale_fill_viridis_c(limits = c(-10, 10), na.value = "white") +
  scale_fill_viridis_c() +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, size=1),
    legend.position='bottom',
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  labs(x=NULL, y=NULL, fill=NULL) +
    guides(fill = guide_colorbar(title.position = "bottom", 
                             direction = "horizontal",
                             frame.colour = "black",
                             barwidth = 20,
                             barheight = 1)) +
  NULL


# Model data check

date_ = sample(mdat$date, 1)
# "date"       "lat_lon"    "SWE"        "lat"        "lon"        "prism_grid" "snow"       "tmean"     
# "tmax"       "tmin"       "ppt"        "gridNumber" "aso_date"   "elevation"  "slope"      "aspect"    
# "year"       "month"      "lat_x_lon"  "nlcd_grid"  "landcover" 

p1 = ggplot(mdat, aes(lon, lat, color=SWE)) + 
  theme_minimal(15) +
  scale_fill_viridis_c() +
  # geom_sf(data=sdat, fill=NA, inherit.aes = FALSE) +
  # geom_tile() +
  geom_point() +
  labs(x=NULL, y=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill=NA, linewidth=1), 
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank()) +
  facet_wrap(~date)
  NULL

  
ggsave(plot=p1, filename="~/Projects/test1.png")

#






# Prediction checks
dat = read_parquet(paste0("~/Projects/ASO-M4/data/", waterbasin, "/predictions/NN-ASO-SWE-Apr01_1981_2021.parquet"))

head(dat)

ggplot(dat, aes(lon, lat, fill=swe_pred)) + geom_tile() + scale_fill_viridis_c() + facet_wrap(~year)

ggplot(filter(dat, swe_pred > 0), aes(lon, lat, fill=swe_pred)) + geom_tile() + scale_fill_viridis_c() + facet_wrap(~year)

ggsave("~/Projects/ASO-M4/figures/test.pdf")

dat = read_parquet("~/Projects/ASO-M4/data/Tuolumne_Watershed/predictions/NN-ASO-SWE-1981-2009_Apr01_V2.parquet")
dat = read_parquet("~/Projects/ASO-M4/data/Tuolumne_Watershed/predictions/NN-ASO-SWE-2010-2021_Apr01_V2.parquet")

ggplot(dat, aes(lon, lat, color=swe_pred)) +
  geom_point() + 
  scale_color_viridis_c() + 
  theme_minimal() +
  labs(x=NULL, y=NULL, fill="SWE Pred") +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  guides(fill = guide_colorbar(title.position = "top", 
                         direction = "vertical",
                         frame.colour = "black",
                         barwidth = 1,
                         barheight = 20)) +
  facet_wrap(~year) +
  NULL

ggsave("~/Projects/ASO-M4/figures/Maps-1981-2009-backpredictions.png", device = "png")


dat1 = read_parquet("~/Projects/ASO-M4/data/Tuolumne_Watershed/predictions/NN-ASO-SWE-1981-2009_Apr01_V2.parquet")
dat2 = read_parquet("~/Projects/ASO-M4/data/Tuolumne_Watershed/predictions/NN-ASO-SWE-2010-2021_Apr01_V2.parquet")

dat = rbind(dat1, dat2)

ggplot(dat, aes(elevation, swe_pred, color=factor(year))) + geom_point() 

dat$lat = round(dat$lat, 1)
dat$lon = round(dat$lon, 1)
dat$lat_lon = paste0(dat$lat, "_", dat$lon)

tdat = dat %>% group_by(lat_lon, year) %>% summarise(swe_pred = mean(swe_pred))

ggplot(tdat, aes(year, swe_pred, group=lat_lon), color='grey') +
  geom_line() +
  scale_color_viridis_c() + 
  theme_minimal() +
  labs(x=NULL, y=NULL, fill="SWE Pred") +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  # guides(fill = guide_colorbar(title.position = "top", 
  #                        direction = "vertical",
  #                        frame.colour = "black",
  #                        barwidth = 1,
  #                        barheight = 20)) +
  # facet_wrap(~year) +
  NULL





dat3 = filter(dat2, year == 2015)

ggplot(dat3, aes(lon, lat)) + geom_tile()
