use std::collections::HashMap;

use log;
use imageproc::definitions::Position;
use serde::{Deserialize, Serialize};
use image::{self, Luma};
use nalgebra as na;


use crate::ext::Frame;
use crate::image_operations::*;
use crate::patch::MatchingCost;
use crate::viewer::FeatureTrackerViewer;
use crate::types::*;


mod feature_detection;
mod feature_tracking;





#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureTrackingConfig {
    pub nlevels: usize, // Number of pyramid levels
    pub ratio: f64, // Ratio of size between each levels

    pub preprocessing_blur: bool,
    pub preprocessing_blur_sigma: Float,

    pub detection_threshold: Float,
    pub detection_min_dist: u32,
    pub detection_blur: Float,

    pub optical_flow_max_iter: usize,
    pub optical_flow_lm_lambda: Float
}


#[derive(Debug, Clone)]
pub struct Feature {
    /// Unique identifier of this feature (within the current frame or globally).
    pub feature_id: usize,

    /// Pixel coordinate in the left image (u, v).
    pub central_point: na::Point2<Float>,
}


pub struct FeatureTracker<'a> {
    config: FeatureTrackingConfig,
    previous_frame_pyramid: Option<(Frame, Pyramid)>,
    viewer: Option<&'a dyn FeatureTrackerViewer>,
    last_keypoint_id: usize
}

impl<'a> FeatureTracker<'a> {
    pub fn new(
        config: FeatureTrackingConfig, 
        viewer: Option<&'a dyn FeatureTrackerViewer>
    ) -> Self {
        FeatureTracker { 
            config, 
            previous_frame_pyramid: None, 
            viewer,
            last_keypoint_id: 0
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.last_keypoint_id;
        self.last_keypoint_id += 1;
        id
    }

    pub fn process_frame(&mut self, in_image: &FloatGrayImage, mut frame: Frame)
    {   

        if let Some(v) = self.viewer {
            v.set_frame(frame.frame_id);
            v.log_image_raw(&in_image.clone().into(), "image/raw");
        }

        
        
        let pyramid = build_image_pyramid(
            in_image, 
            self.config.nlevels, 
            self.config.ratio,
            self.config.preprocessing_blur,
            self.config.preprocessing_blur_sigma
        );

        if let Some(v) = self.viewer && log::max_level() >= log::LevelFilter::Debug {
            let dynimage_pyramid = pyramid.iter()
                .map(|level_img| level_img.clone().into())
                .collect::<Vec<_>>();
            let prev_pyr = self.previous_frame_pyramid.as_ref().map(|(_, prev_pyr)| prev_pyr);
            let dynimage_prev_pyr = prev_pyr.map(|p| 
                p.iter()
                 .map(|level_img| level_img.clone().into())
                 .collect::<Vec<_>>()
            );
            
            v.log_image_pyramid(&dynimage_pyramid, (&dynimage_prev_pyr).as_ref().map(|p| &**p), "pyramid");
            let img_fine = pyramid[0].clone().into();
            v.log_image_raw(&img_fine, "image/preprocessed");

        }




        if let Some((previous_frame, previous_pyramid)) = self.previous_frame_pyramid.as_ref() 
        {
            let mut transform_maps = HashMap::new();
            feature_tracking::track_points(
                previous_pyramid, 
                &pyramid, 
                &previous_frame,
                &mut transform_maps, 
                self.config.optical_flow_max_iter,
                self.config.optical_flow_lm_lambda,
                MatchingCost::SSD,
                self.viewer
            );
            
            let tracked_features: Vec<Feature> = previous_frame.features.iter()
                .filter_map(|feature| {
                    transform_maps
                        .get(&feature.feature_id)
                        .and_then(|transform| Some(
                            Feature {
                                feature_id: feature.feature_id,
                                central_point: transform * feature.central_point
                            }
                        ))
                })
                .collect();

            println!("tracked features: {}", tracked_features.len());
            frame.features.extend(tracked_features);


            // if let Some(v) = self.viewer {
            //     let coords: Vec<[f32; 2]> = tracked_points.iter().map(|feature| feature.pixel_coord.into()).collect();
            //     let ids: Vec<String> = tracked_points.iter().map(|feature| format!("{}", feature.feature_id)).collect();
            //     v.log_features(&coords, "image/old_features", Some(&ids), Some(rerun::Color::from_rgb(0, 255, 0)));
            //     v.log_image_raw(&previous_pyramid[0].clone().into(), "image/old_image");
            // }

        };

        



        let new_corners = feature_detection::add_points(
            &pyramid, 
            &frame.features,
            self.config.detection_threshold, 
            self.config.detection_min_dist,
            self.config.detection_blur,
            self.viewer
        );

        let new_features = new_corners
            .iter()
            .map(|corner| 
                Feature { 
                    feature_id: self.next_id(), 
                    central_point: na::Point2::new(corner.x() as f32, corner.y() as f32)
                });

        frame.features.extend(new_features);

        if let Some(v) = self.viewer {
            let coords = frame.features.iter().map(|f| f.central_point.coords.into()).collect::<Vec<_>>();
            let labels = frame.features.iter().map(|f| format!("{}", f.feature_id % 1000)).collect::<Vec<_>>();
            v.log_features(&coords, "image/features", Some(&labels), None);
        }

        self.previous_frame_pyramid = Some((frame, pyramid));
    }




    pub fn get_pyramid(&self) -> Option<&Pyramid> 
    {
        self.previous_frame_pyramid.as_ref().map(|(_, pyr)| pyr)
    }
}


