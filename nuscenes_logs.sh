#!/usr/bin/env bash

conda activate open-mmlab

# Compute AP_dist_values for each attribute detection

# overall mAP
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/mAP \
--legend mAP \
--title "Overall mAP Curve" --out  zzz_pdf_folder/overall_map_curve.pdf

# cycle.with_rider
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/cycle.with_rider_AP_dist_0.5 pts_bbox_NuScenes/cycle.with_rider_AP_dist_1.0 pts_bbox_NuScenes/cycle.with_rider_AP_dist_2.0 pts_bbox_NuScenes/cycle.with_rider_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "cycle_with_rider AP Curves" --out  zzz_pdf_folder/cycle_with_rider_AP_Curves.pdf

# cycle.without_rider
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/cycle.without_rider_AP_dist_0.5 pts_bbox_NuScenes/cycle.without_rider_AP_dist_1.0 pts_bbox_NuScenes/cycle.without_rider_AP_dist_2.0 pts_bbox_NuScenes/cycle.without_rider_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "cycle_without_rider AP Curves" --out  zzz_pdf_folder/cycle_without_rider_AP_Curves.pdf

# pedestrian.moving
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/pedestrian.moving_AP_dist_0.5 pts_bbox_NuScenes/pedestrian.moving_AP_dist_1.0 pts_bbox_NuScenes/pedestrian.moving_AP_dist_2.0 pts_bbox_NuScenes/pedestrian.moving_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "pedestrian_moving AP Curves" --out  zzz_pdf_folder/pedestrian_moving_AP_Curves.pdf

# pedestrian.standing
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/pedestrian.standing_AP_dist_0.5 pts_bbox_NuScenes/pedestrian.standing_AP_dist_1.0 pts_bbox_NuScenes/pedestrian.standing_AP_dist_2.0 pts_bbox_NuScenes/pedestrian.standing_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "pedestrian_standing AP Curves" --out  zzz_pdf_folder/pedestrian_standing_AP_Curves.pdf

# pedestrian.sitting_lying_down
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/pedestrian.sitting_lying_down_AP_dist_0.5 pts_bbox_NuScenes/pedestrian.sitting_lying_down_AP_dist_1.0 pts_bbox_NuScenes/pedestrian.sitting_lying_down_AP_dist_2.0 pts_bbox_NuScenes/pedestrian.sitting_lying_down_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "pedestrian_sitting_lying_down AP Curves" --out  zzz_pdf_folder/pedestrian_sitting_lying_down_AP_Curves.pdf

# vehicle.moving
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/vehicle.moving_AP_dist_0.5 pts_bbox_NuScenes/vehicle.moving_AP_dist_1.0 pts_bbox_NuScenes/vehicle.moving_AP_dist_2.0 pts_bbox_NuScenes/vehicle.moving_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "vehicle_moving AP Curves" --out  zzz_pdf_folder/vehicle_moving_AP_Curves.pdf

# vehicle.parked
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/vehicle.parked_AP_dist_0.5 pts_bbox_NuScenes/vehicle.parked_AP_dist_1.0 pts_bbox_NuScenes/vehicle.parked_AP_dist_2.0 pts_bbox_NuScenes/vehicle.parked_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "vehicle_parked AP Curves" --out  zzz_pdf_folder/vehicle_parked_AP_Curves.pdf

# vehicle.stopped
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--mode eval --keys pts_bbox_NuScenes/vehicle.stopped_AP_dist_0.5 pts_bbox_NuScenes/vehicle.stopped_AP_dist_1.0 pts_bbox_NuScenes/vehicle.stopped_AP_dist_2.0 pts_bbox_NuScenes/vehicle.stopped_AP_dist_4.0 \
--legend AP_dist_0.5 AP_dist_1.0 AP_dist_2.0 AP_dist_4.0 \
--title "vehicle_stopped AP Curves" --out  zzz_pdf_folder/vehicle_stopped_AP_Curves.pdf

# Compute loss curves
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--keys loss --title "Overall Loss Curve" \
--legend loss \
--out  zzz_pdf_folder/overall_loss_curves.pdf

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--keys task6.loss_heatmap task7.loss_heatmap task8.loss_heatmap --legend cycle_heatmap_loss pedestrian_heatmap_loss vehicle_heatmap_loss --title "Attribute Heatmap Losses" \
--out  zzz_pdf_folder/attribute_heatmap_losses.pdf

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_mini_batch4_10epochs/20210501_225608.log.json \
--keys task6.loss_bbox task7.loss_bbox task8.loss_bbox --legend cycle_bbox_loss pedestrian_bbox_loss vehicle_bbox_loss --title "Attribute Bounding Box Losses" \
--out  zzz_pdf_folder/attribute_bbox_losses.pdf

# zip all files and put in directory
zip -r pdf_curves.zip zzz_pdf_folder/