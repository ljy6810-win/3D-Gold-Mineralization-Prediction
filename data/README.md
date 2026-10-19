本目录只说明程序要读取的文件名。钻孔、化验、岩性、剖面、地表高程和由此算出的概率场、距离场都不在本仓库中。

请把自有数据放在本目录后再运行，不要把这些文件提交到公开仓库。

概率场读取：

- lithology_samples.csv
- surface_dem_grid.csv

距离场读取：

- parsed_geology.xlsx
- section_params.csv
- ../results/lithology_probability.csv

三模型训练读取：

- feature_table.csv
- voxel_hole_id.csv
- feature_cache/
- cnn_volume.dat
