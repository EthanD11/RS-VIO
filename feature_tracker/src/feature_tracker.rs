use imageproc::gradients;
use serde::{Deserialize, Serialize};
use image::{self, GrayImage, ImageBuffer, Luma, Pixel};

use crate::ext::Frame;
use crate::image_operations::*;
use crate::viewer::FeatureTrackerViewer;
use crate::types::*;





#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureTrackingConfig {
    pub nlevels: usize, // Number of pyramid levels
    pub ratio: f64, // Ratio of size between each levels
    pub preprocessing_blur: bool,
    pub preprocessing_blur_sigma: Float
}


#[derive(Debug, Clone)]
pub struct Feature {
    /// Unique identifier of this feature (within the current frame or globally).
    pub feature_id: usize,

    /// Pixel coordinate in the left image (u, v).
    pub pixel_coord: [f32; 2],

    /// Undistorted pixel coordinate (u, v). `[-1, -1]` means invalid.
    pub undistorted_coord: [f32; 2],
}


pub struct FeatureTracker<'a> {
    config: FeatureTrackingConfig,
    previous_pyramid: Option<Pyramid>,
    viewer: Option<&'a dyn FeatureTrackerViewer>
}

impl<'a> FeatureTracker<'a> {
    pub fn new(
        config: FeatureTrackingConfig, 
        viewer: Option<&'a dyn FeatureTrackerViewer>
    ) -> Self {
        FeatureTracker { config, previous_pyramid: None, viewer }
    }

    pub fn process_frame(&mut self, in_image: &GrayImage, frame: &mut Frame) 
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
            let dynimage_pyramid = pyramid.iter().map(|l| l.clone().into()).collect::<Vec<_>>();
            v.log_image_pyramid(&dynimage_pyramid.iter().collect::<Vec<_>>(), "pyramid");
        }

        if let Some(_previous_pyramid) = self.previous_pyramid.as_ref() {
            println!("there was a previous pyramid");
        };

        feature_detection::add_points(&pyramid, frame);


        self.previous_pyramid = Some(pyramid);
    }

    pub fn get_pyramid(&self) -> Option<&Pyramid> {
        self.previous_pyramid.as_ref()
    }
}



// Image pyramid builder




// Feature detection methods
pub mod feature_detection {
    use super::*;

    pub fn add_points(pyramid: &Pyramid, frame: &mut Frame) {
        let image_fine = pyramid.first().unwrap();
        // gradients::gradients(image_fine, horizontal_kernel, vertical_kernel, f);
        todo!()
    }

    // fn criterion(in_image: &ImageBuffer<Luma<Float>, Vec<Float>>, patch: )
}





#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_image_pyramid() {
        let (w, h) = (120, 60);
        let gray_image = GrayImage::from_vec(w, h, 
            Vec::from_iter(std::iter::repeat(5).take((w*h) as usize))).unwrap();

        let pyramid = build_image_pyramid(&gray_image, 3, 2.0, false, 0.0);
        assert!(pyramid[0].dimensions() == (120, 60));
        assert!(pyramid[1].dimensions() == (60, 30));
        assert!(pyramid[2].dimensions() == (30, 15));
    }
}
