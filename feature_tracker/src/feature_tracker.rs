use std::ops::Deref;

use imageproc::gradients;
use serde::{Deserialize, Serialize};
use image::{self, DynamicImage, GrayImage, ImageBuffer, Luma, Pixel, imageops::*};

use crate::ext::Frame;
use crate::viewer::FeatureTrackerViewer;


type ImageFloat = f32;
type Pyramid = Vec<ImageBuffer<Luma<ImageFloat>, Vec<ImageFloat>>>;


#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureTrackingConfig {
    pub nlevels: usize, // Number of pyramid levels
    pub ratio: f64, // Ratio of size between each levels
    pub preprocessing_blur: bool,
    pub preprocessing_blur_sigma: ImageFloat
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

    pub fn process_frame(&mut self, input_image: &GrayImage, frame: &mut Frame) 
    {   

        if let Some(v) = self.viewer {
            v.set_frame(frame.frame_id);
            v.log_image_raw(&input_image.clone().into(), "image/raw");
        }

        
        
        let pyramid = build_image_pyramid(
            input_image, 
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

        add_points(&pyramid, frame);


        self.previous_pyramid = Some(pyramid);
    }

    pub fn get_pyramid(&self) -> Option<&Pyramid> {
        self.previous_pyramid.as_ref()
    }
}



fn build_image_pyramid(input_image: &GrayImage, nlevels: usize, ratio: f64, bluring: bool, blur_sigma: ImageFloat) -> Pyramid {
    let float_image: ImageBuffer<Luma<ImageFloat>, Vec<_>> = ImageBuffer::from_vec(
        input_image.width(), 
        input_image.height(),
        input_image.pixels()
            .map(|p| (p.channels()[0] as ImageFloat) / 255.0)
            .collect::<Vec<_>>()
    ).unwrap();

    let float_image = if bluring {
        blur(&float_image, blur_sigma.into())
    } else {
        float_image
    };

    let (w0, h0) = float_image.dimensions();

    let mut pyramid = Vec::new();
    pyramid.push(float_image);
    for l in 1..nlevels {

        let level = i32::try_from(l).expect("Overflow caused by number of pyramid levels superior to i32::MAX. Consider decreasing number of pyramid levels");
        let nwidth = (f64::try_from(w0).unwrap() / ratio.powi(level)).round() as u32;
        let nheight = (f64::try_from(h0).unwrap() / ratio.powi(level)).round() as u32;
        
        let previous_image = pyramid.last().unwrap();
        pyramid.push(resize(previous_image, nwidth, nheight, Triangle));
    } 
    pyramid
}


fn add_points(pyramid: &Pyramid, frame: &mut Frame) {
    let image_fine = pyramid.first().unwrap();
    gradients::gradients(image_fine, horizontal_kernel, vertical_kernel, f);
    todo!()
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
