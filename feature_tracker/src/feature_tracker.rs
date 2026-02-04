use std::collections::HashMap;

use log;
use imageproc::definitions::Position;
use serde::{Deserialize, Serialize};
use image::{self, GrayImage, ImageBuffer, Luma, Pixel};
use nalgebra as na;


use crate::ext::Frame;
use crate::image_operations::*;
use crate::viewer::FeatureTrackerViewer;
use crate::types::*;


pub mod feature_detection;





#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureTrackingConfig {
    pub nlevels: usize, // Number of pyramid levels
    pub ratio: f64, // Ratio of size between each levels
    pub preprocessing_blur: bool,
    pub preprocessing_blur_sigma: Float,
    pub detection_threshold: Float,
    pub detection_min_dist: u32,
    pub detection_blur: Float
}


#[derive(Debug, Clone)]
pub struct Feature {
    /// Unique identifier of this feature (within the current frame or globally).
    pub feature_id: usize,

    /// Pixel coordinate in the left image (u, v).
    pub pixel_coord: [f32; 2],
}


pub struct FeatureTracker<'a> {
    config: FeatureTrackingConfig,
    previous_pyramid: Option<Pyramid>,
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
            previous_pyramid: None, 
            viewer,
            last_keypoint_id: 0
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.last_keypoint_id;
        self.last_keypoint_id += 1;
        id
    }

    pub fn process_frame(&mut self, in_image: &FloatGrayImage, frame: &mut Frame) 
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
        if let Some(v) = self.viewer {
            let dynimage_pyramid = pyramid.iter().map(|level_img| level_img.clone().into()).collect::<Vec<_>>();
            v.log_image_pyramid(&dynimage_pyramid.iter().collect::<Vec<_>>(), "pyramid");
            let img_fine = pyramid[0].clone().into();
            v.log_image_raw(&img_fine, "image/preprocessed");
        }




        if let Some(_previous_pyramid) = self.previous_pyramid.as_ref() {
            println!("there was a previous pyramid");
        };

        



        let new_corners = feature_detection::add_points(
            &pyramid, 
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
                    pixel_coord: [corner.x() as f32, corner.y() as f32]
                });

        frame.features.extend(new_features);

        if let Some(v) = self.viewer {
            let coords = frame.features.iter().map(|f| f.pixel_coord).collect::<Vec<_>>();
            let labels = frame.features.iter().map(|f| format!("{}", f.feature_id)).collect::<Vec<_>>();
            v.log_features(&coords, "image/features", Some(&labels));
        }


        self.previous_pyramid = Some(pyramid);
    }

    pub fn get_pyramid(&self) -> Option<&Pyramid> {
        self.previous_pyramid.as_ref()
    }
}


fn track_points(
    pyramid0: &Pyramid,
    pyramid1: &Pyramid,
    transform_maps0: HashMap<usize, na>
)






