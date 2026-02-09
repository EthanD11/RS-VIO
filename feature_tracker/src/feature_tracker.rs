use std::collections::HashMap;
use std::ops::Not;
use std::option;

use log;
use imageproc::definitions::Position;
use rerun::TransformRelation;
use serde::{Deserialize, Serialize};
use image::{self, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel};
use nalgebra as na;


use crate::ext::Frame;
use crate::feature_tracker::feature_detection::ShiTomasiCorner;
use crate::image_operations::*;
use crate::patch::{Patch, Patch52};
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
    pub detection_blur: Float,

    pub optical_flow_max_iter: usize,
    pub optical_flow_lm_lambda: Float
}


#[derive(Debug, Clone)]
pub struct Feature {
    /// Unique identifier of this feature (within the current frame or globally).
    pub feature_id: usize,

    /// Pixel coordinate in the left image (u, v).
    pub pixel_coord: na::Point2<Float>,
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
            track_points(
                previous_pyramid, 
                &pyramid, 
                &previous_frame,
                &mut transform_maps, 
                self.config.optical_flow_max_iter,
                self.config.optical_flow_lm_lambda,
                self.viewer
            );
            
            let tracked_features: Vec<Feature> = previous_frame.features.iter()
                .filter_map(|feature| {
                    transform_maps
                        .get(&feature.feature_id)
                        .and_then(|transform| Some(
                            Feature {
                                feature_id: feature.feature_id,
                                pixel_coord: transform * feature.pixel_coord
                            }
                        ))
                })
                .collect();

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
                    pixel_coord: na::Point2::new(corner.x() as f32, corner.y() as f32)
                });

        frame.features.extend(new_features);

        if let Some(v) = self.viewer {
            let coords = frame.features.iter().map(|f| f.pixel_coord.coords.into()).collect::<Vec<_>>();
            let labels = frame.features.iter().map(|f| format!("{}", f.feature_id)).collect::<Vec<_>>();
            v.log_features(&coords, "image/features", Some(&labels), None);
        }

        self.previous_frame_pyramid = Some((frame, pyramid));
    }

    pub fn get_pyramid(&self) -> Option<&Pyramid> {
        self.previous_frame_pyramid.as_ref().map(|(_, pyr)| pyr)
    }
}


fn track_points(
    pyramid0: &Pyramid,
    pyramid1: &Pyramid,
    frame0: &Frame,
    transform_maps0: &mut HashMap<usize, na::Isometry2<Float>>,
    max_iteration: usize,
    lm_lambda: Float,
    viewer: Option<&dyn FeatureTrackerViewer>
) {

    for feature in frame0.features.iter() {
        let tracking_result = track_point(
            pyramid0, 
            pyramid1, 
            feature, 
            max_iteration, 
            lm_lambda,
            viewer
        );
        if let Ok(transform) = tracking_result {
            transform_maps0.insert(feature.feature_id, transform);
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum TrackPointError {
    OutOfBound
}


fn track_point(
    pyramid0: &Pyramid,
    pyramid1: &Pyramid,
    feature: &Feature,
    max_iteration: usize,
    lm_lambda: Float,
    viewer: Option<&dyn FeatureTrackerViewer>
) -> Result<na::Isometry2<Float>, TrackPointError>
{


    let (w, h) = pyramid0.first().unwrap().dimensions();
    let (w, h) = (w as Float, h as Float);

    let fx = feature.pixel_coord.x;
    let fy = feature.pixel_coord.y;

    let mut transform = na::Isometry2::identity();
    for level in (0..pyramid0.len()).rev()
    {   
        
        let img0 = pyramid0.get(level).unwrap();
        let img1 = pyramid1.get(level).unwrap();
        let (w_lvl, h_lvl) = img0.dimensions();

        let scaling_x = w_lvl as Float / w;
        let scaling_y = h_lvl as Float / h;
        let level_coords = na::Point2::new(
            scaling_x*(fx + 0.5) - 0.5, 
            scaling_y*(fy + 0.5) - 0.5
        );

        transform = track_point_at_level(
            img0, 
            img1, 
            &level_coords, 
            &transform, 
            max_iteration,
            lm_lambda,
            level,
            feature.feature_id,
            viewer
        )?;

        // Scale to the next pyramid level
        if level > 0 {
            let (w_next, h_next) = pyramid0.get(level-1).unwrap().dimensions();
            transform.translation.vector[0] *= w_next as Float / w_lvl as Float;
            transform.translation.vector[1] *= h_next as Float / h_lvl as Float;
        }

    }
    
    Ok(transform)
}



fn track_point_at_level(
    img0: &FloatGrayImage,
    img1: &FloatGrayImage,
    feature_coords: &na::Point2<Float>,
    initial_guess: &na::Isometry2<Float>,
    max_iteration: usize,
    lm_lambda: Float,
    level: usize,
    feature_id: usize,
    viewer: Option<&dyn FeatureTrackerViewer>
) -> Result<na::Isometry2<Float>, TrackPointError>
{
    let mut transform = initial_guess.clone();

    let patch = Patch52::new(img0, feature_coords.coords.as_ref(), lm_lambda);
    let jac = patch.jacobian();
    let inverse_hessian = patch.inverse_hessian();

    if let Some(v) = viewer && log::max_level() >= log::LevelFilter::Debug {
        v.log_features(
            &patch.shifted_indexes(&transform), 
            &format!("pyramid/level{level}/features/{feature_id}/patch"), None, None);
    }

    let mut iterates = vec![(transform*feature_coords).into()];
    for i in 0..max_iteration 
    {
        log::debug!("Feature id: {feature_id}");
        log::debug!("iter: {i}");

        
        

        let residuals = patch.residuals(img1, &transform);
        log::debug!("{}\n", residuals.norm());

        let b = jac.transpose() * residuals;
        let twist = inverse_hessian * b;
        transform *= exp_se2(&(-twist));

        let iterate = transform * feature_coords;
        if viewer.is_some() {
            iterates.push(iterate.into());
        }
        if in_bounds(img0, iterate.x, iterate.y).not()
        {
            return Err(TrackPointError::OutOfBound)
        }

        if twist.norm() < 1e-3 {
            break
        }
    }

    if let Some(v) = viewer && log::max_level() >= log::LevelFilter::Debug {
        let labels = (0..iterates.len()).map(|i| format!("{i}")).collect::<Vec<_>>();
        v.log_features(
            &iterates, 
            &format!("pyramid/level{level}/features/{feature_id}/iterates"), 
            Some(&labels), 
            None
        );
    }

    Ok(transform)
}


fn exp_se2(twist: &na::Vector3<Float>) ->  na::Isometry2<Float> {
    let theta = twist[0];
    let v = twist.fixed_rows::<2>(1);
    
    let (diag, cross_diag) = if theta.abs() > 1e-4 {
        (
            theta.sin() / theta, 
            (1.0 - theta.cos()) / theta
        )
    } else {
        let theta_sq = theta*theta;
        (
            1.0 - theta.powi(2)/6.0,
            (0.5 - theta_sq / 24.0)*theta
        )
    };

    let translation = na::Vector2::new(
        diag*v[0] - cross_diag*v[1],
        cross_diag*v[0] + diag*v[1]
    );

    na::Isometry2::<Float>::new(translation, theta)

}


fn log_se2(transformation: &na::Isometry2<Float>) -> na::Vector3<Float>
{
    let t_matrix = transformation.to_matrix();
    let theta = Float::atan2(t_matrix[(1, 0)], t_matrix[(0, 0)]);
    let diag = if theta.abs() > 1e-3 {
        let diag = theta * theta.sin() / (2.0 * (1.0 - theta.cos()));
        if !diag.is_finite() {
            println!("theta: {theta:?}");
        }
        assert!(diag.is_finite());
        diag
    } else {
        let theta_sq = theta.powi(2);
        let diag = (1.0 - theta_sq / 6.0) / (1.0 - theta_sq  / 12.0);
        assert!(diag.is_finite());
        diag
    };
    let cross_diag = 0.5*theta;
    let translation = transformation.translation.vector;
    let v1 = diag * translation[0] + cross_diag * translation[1];
    let v2 = -cross_diag * translation[0] + diag * translation[1];
    na::Vector3::new(theta, v1, v2)
}

#[cfg(test)]
mod test {

    use super::*;

    fn check_se2_log(twist: &na::Vector3<f32>) {
        let pi = std::f32::consts::PI;

        let transformation = exp_se2(twist);
        let mut twist_prime = log_se2(&transformation);
        let mut tau = twist.to_owned();

        tau[0] = tau[0] - tau[0].div_euclid(2.0*pi) * 2.0*pi;
        twist_prime[0] = twist_prime[0] - twist_prime[0].div_euclid(2.0*pi) * 2.0*pi;

        assert!(twist.relative_eq(&twist_prime, 1e-6, 1e-6));
    }

    #[test]
    fn test_exp_se2() {
        let pi = std::f32::consts::PI;
        let tol = 1e-6;

        let twist = na::Vector3::<Float>::new(0.0, 0.0, 0.0);
        let transformation = exp_se2(&twist);
        assert!(transformation.rotation.to_rotation_matrix().matrix().relative_eq(&na::Matrix2::<Float>::from_row_slice(&[1.0, 0.0, 0.0, 1.0]), tol, tol));
        assert!(transformation.translation.vector.relative_eq(&Vec2::new(0.0, 0.0), tol, tol));

        
        let twist = na::Vector3::<Float>::new(pi, 0.0, 0.0);
        let transformation = exp_se2(&twist);
        assert!(transformation.rotation.to_rotation_matrix().matrix().relative_eq(&na::Matrix2::<Float>::from_row_slice(&[-1.0, 0.0, 0.0, -1.0]), tol, tol));
        assert!(transformation.translation.vector.relative_eq(&Vec2::new(0.0, 0.0), tol, tol));

        let twist = na::Vector3::<Float>::new(0.0, 1.32, -1.56);
        let transformation = exp_se2(&twist);
        assert!(transformation.rotation.to_rotation_matrix().matrix().relative_eq(&na::Matrix2::<Float>::from_row_slice(&[1.0, 0.0, 0.0, 1.0]), tol, tol));
        assert!(transformation.translation.vector.relative_eq(&Vec2::new(1.32, -1.56), tol, tol));

        check_se2_log(&na::Vector3::new(0.0, 0.0, 0.0));
        check_se2_log(&na::Vector3::new(pi / 3.0, 1.0, 2.5));
        check_se2_log(&na::Vector3::new(1.795, 1.0, 2.5));
        check_se2_log(&na::Vector3::new(1e-8, 1.0, 2.5));
        check_se2_log(&na::Vector3::new(1e-7, 1.0, 2.5));
        check_se2_log(&na::Vector3::new(2e-7, 1.0, 2.5));
    }
}







