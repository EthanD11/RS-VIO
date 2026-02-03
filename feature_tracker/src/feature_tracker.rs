use imageproc::definitions::Position;
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
            v.log_features(&frame.features.iter().map(|f| f.pixel_coord).collect::<Vec<_>>(), "image/features");
        }


        self.previous_pyramid = Some(pyramid);
    }

    pub fn get_pyramid(&self) -> Option<&Pyramid> {
        self.previous_pyramid.as_ref()
    }
}



// Image pyramid builder




// Feature detection methods
pub mod feature_detection {
    use std::{cmp, ops::Not};

    use image::{GenericImage, imageops::fast_blur};
    use imageproc::{filter, definitions::{Position, Score}, suppress};

    use super::*;

    #[derive(Copy, Clone, Debug)]
    pub struct ShiTomasiCorner {
        x: u32,
        y: u32,
        score: f32,
        from_past_feature: bool
    }
    impl ShiTomasiCorner {
        fn new(x: u32, y: u32, score: f32) -> ShiTomasiCorner {
            ShiTomasiCorner {x, y, score, from_past_feature: false}
        }

        fn from_feature(f: &Feature, score: f32) -> ShiTomasiCorner {
            ShiTomasiCorner { 
                x: f.pixel_coord[0].round() as u32, 
                y: f.pixel_coord[1].round() as u32, 
                score, 
                from_past_feature: true
            }
        }
    }
    impl Position for ShiTomasiCorner {
        #[inline]
        fn x(&self) -> u32 {
            self.x
        }
        #[inline]
        fn y(&self) -> u32 {
            self.y
        }
    }
    impl Score for ShiTomasiCorner {
        #[inline]
        fn score(&self) -> f32 {
            self.score
        }
    }

    pub fn add_points(
        pyramid: &Pyramid,
        threshold: Float, 
        min_dist_between_points: u32,
        detection_blur: Float,
        viewer: Option<&dyn FeatureTrackerViewer>
    ) -> Vec<ShiTomasiCorner>  
    {
        let image_fine = pyramid.first().unwrap();
        
        let score_map = shi_tomasi_score(&image_fine, detection_blur, viewer);
        let corners = suppress_non_maximum(&score_map, min_dist_between_points, threshold);
        
        corners
            .into_iter()
            .filter(|c| c.from_past_feature.not())
            .collect()
    }

    fn shi_tomasi_score(
        in_image: &FloatGrayImage,
        detection_blur: Float,
        viewer: Option<&dyn FeatureTrackerViewer>
    ) -> FloatGrayImage
    {
        let (w, h) = in_image.dimensions();

        let kernel = [-1.0, 0.0, 1.0];
        // let kernel = Kernel::new(&kernel_data, 3, 1);
        let mut dimage_dxx = filter::horizontal_filter(&in_image, &kernel);
        let mut dimage_dyy = filter::vertical_filter(&in_image, &kernel);

        if let Some(v) = viewer {
            v.log_map(dimage_dxx.as_raw(), w, h, None, "image/shi_tomasi/grad/x");
            v.log_map(dimage_dyy.as_raw(), w, h, None, "image/shi_tomasi/grad/y");
        }

        let mut dimage_dxy = FloatGrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let &dx = unsafe { dimage_dxx.as_raw().get_unchecked((y*w+x) as usize) };
                let &dy = unsafe { dimage_dyy.as_raw().get_unchecked((y*w+x) as usize) };

                unsafe { dimage_dxx.unsafe_put_pixel(x, y, Luma([dx*dx])) };
                unsafe { dimage_dyy.unsafe_put_pixel(x, y, Luma([dy*dy])) };
                unsafe { dimage_dxy.unsafe_put_pixel(x, y, Luma([dx*dy])) };
            }
        }

        if let Some(v) = viewer {
            v.log_map(dimage_dxx.as_raw(), w, h, None, "image/shi_tomasi/structure/dxx");
            v.log_map(dimage_dyy.as_raw(), w, h, None, "image/shi_tomasi/structure/dyy");
            v.log_map(dimage_dxy.as_raw(), w, h, None, "image/shi_tomasi/structure/dxy");
        }

        let dimage_dxx = fast_blur(&dimage_dxx, detection_blur);
        let dimage_dyy = fast_blur(&dimage_dyy, detection_blur);
        let dimage_dxy = fast_blur(&dimage_dxy, detection_blur);

        if let Some(v) = viewer {
            v.log_map(dimage_dxx.as_raw(), w, h, None, "image/shi_tomasi/structure_smooth/dxx");
            v.log_map(dimage_dyy.as_raw(), w, h, None, "image/shi_tomasi/structure_smooth/dyy");
            v.log_map(dimage_dxy.as_raw(), w, h, None, "image/shi_tomasi/structure_smooth/dxy");
        }

        let mut score_map = FloatGrayImage::new(w, h);

        for y in 0..h {
            for x in 0..w {
                let i = (y*w+x) as usize;

                let &dxx = unsafe { dimage_dxx.as_raw().get_unchecked(i) };
                let &dyy = unsafe { dimage_dyy.as_raw().get_unchecked(i) }; 
                let &dxy = unsafe { dimage_dxy.as_raw().get_unchecked(i) }; 

                let trace = dxx + dyy;
                let det = dxx*dyy - dxy*dxy;

                let delta = Float::max(trace*trace - 4.0*det, 0.0);

                let score = 500.0*((trace - delta.sqrt())).abs();
                assert!(score == 0.0 || score.is_normal());

                if let Some(_) = viewer {
                    unsafe {
                        score_map.unsafe_put_pixel(x, y, Luma::<f32>([score]));
                    }
                }
            }
        }

        if let Some(v) = viewer {
            v.log_map(&score_map.as_raw(), w, h, None, "image/shi_tomasi/score_map");
        }
        
        score_map
    }


    /// Code (lightly) adapte form imageproc::suppress::suppress_non_maximum
    /// Returned image has zeroes for all inputs pixels which do not have the greatest
    /// intensity in the (2 * radius + 1) square block centred on them.
    /// Ties are resolved lexicographically.
    pub fn suppress_non_maximum<I>(score_map: &I, radius: u32, threshold: Float) -> Vec<ShiTomasiCorner>
    where
        I: GenericImage<Pixel = Luma<Float>>
    {
        let (width, height) = score_map.dimensions();
        let mut out = Vec::new();
        if width == 0 || height == 0 {
            return out;
        }

        // We divide the image into a grid of blocks of size r * r. We find the maximum
        // value in each block, and then test whether this is in fact the maximum value
        // in the (2r + 1) * (2r + 1) block centered on it. Any pixel that's not maximal
        // within its r * r grid cell can't be a local maximum so we need only perform
        // the (2r + 1) * (2r + 1) search once per r * r grid cell (as opposed to once
        // per pixel in the naive implementation of this algorithm).

        for y in (0..height).step_by(radius as usize + 1) {
            for x in (0..width).step_by(radius as usize + 1) {
                let mut best_x = x;
                let mut best_y = y;
                let mut best_score = score_map.get_pixel(x, y)[0];

                // These mins are necessary for when radius > min(width, height)
                for cy in y..cmp::min(height, y + radius + 1) {
                    for cx in x..cmp::min(width, x + radius + 1) {
                        let ci = unsafe { score_map.unsafe_get_pixel(cx, cy)[0] };
                        if ci < best_score {
                            continue;
                        }
                        if ci > best_score || (cx, cy) < (best_x, best_y) {
                            best_x = cx;
                            best_y = cy;
                            best_score = ci;
                        }
                    }
                }

                if best_score >= threshold {
                    let x0 = best_x.saturating_sub(radius);
                    let x1 = x;
                    let x2 = cmp::min(width, x + radius + 1);
                    let x3 = cmp::min(width, best_x + radius + 1);
    
                    let y0 = best_y.saturating_sub(radius);
                    let y1 = y;
                    let y2 = cmp::min(height, y + radius + 1);
                    let y3 = cmp::min(height, best_y + radius + 1);
    
                    // Above initial r * r block
                    let mut failed = contains_greater_value(score_map, best_x, best_y, best_score, y0, y1, x0, x3);
                    // Left of initial r * r block
                    failed |= contains_greater_value(score_map, best_x, best_y, best_score, y1, y2, x0, x1);
                    // Right of initial r * r block
                    failed |= contains_greater_value(score_map, best_x, best_y, best_score, y1, y2, x2, x3);
                    // Below initial r * r block
                    failed |= contains_greater_value(score_map, best_x, best_y, best_score, y2, y3, x0, x3);
    
                    if !failed {
                        out.push(ShiTomasiCorner::new(x, y, best_score))
                    }
                }

            }
        }

        out
    }

    /// Returns true if the given block contains a larger value than
    /// the input, or contains an equal value with lexicographically
    /// lesser coordinates.
    #[allow(clippy::too_many_arguments)]
    fn contains_greater_value<I, C>(
        image: &I,
        x: u32,
        y: u32,
        v: C,
        y_lower: u32,
        y_upper: u32,
        x_lower: u32,
        x_upper: u32,
    ) -> bool
    where
        I: GenericImage<Pixel = Luma<C>>,
        C: image::Primitive + PartialOrd,
    {
        for cy in y_lower..y_upper {
            for cx in x_lower..x_upper {
                let ci = unsafe { image.unsafe_get_pixel(cx, cy)[0] };
                if ci < v {
                    continue;
                }
                if ci > v || (cx, cy) < (x, y) {
                    return true;
                }
            }
        }
        false
    }


    

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
