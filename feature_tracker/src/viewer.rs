use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel, PixelWithColorType, Primitive};
use image::codecs::jpeg::JpegEncoder;
use crate::types::*;
use rerun::{self as rr};

/// Viewer trait for basic visualization operations
pub trait FeatureTrackerViewer {
    
    /// Log raw image
    fn log_image_raw(&self, image: &DynamicImage, entity_path: &str);
    
    /// Log image with features drawn on it
    fn log_features(&self, features: &[[f32; 2]], entity_path: &str);

    fn log_image_pyramid(&self, pyramid: &[&DynamicImage], entity_path: &str);

    fn log_map(&self, img: &[f32], width: u32, height: u32, cmap: Option<rr::components::Colormap>, entity_path: &str);
    
    /// Log image with features colored by feature ID
    /// Features should be provided as (feature_id, [x, y]) tuples
    fn log_image_with_features_colored(&mut self, image: &[u8], width: u32, height: u32, features: &[(usize, [f32; 2])], entity_path: &str);
    
    /// Set the current frame/timestamp
    fn set_frame(&self, frame_id: i64);
    
}



impl FeatureTrackerViewer for rr::RecordingStream {
    fn log_image_raw(&self, in_image: &DynamicImage, entity_path: &str) {
        // let in_image = in_image.to_luma8();
        let mut jpeg_bytes = Vec::new();
        let mut encoder = JpegEncoder::new(&mut jpeg_bytes);
        encoder.encode_image(in_image).unwrap();
        self.log(entity_path, &rr::EncodedImage::from_file_contents(jpeg_bytes))
            .unwrap();
    }

    fn log_image_pyramid(&self, pyramid: &[&DynamicImage], entity_path: &str) 
    {
        for (l, image_l) in pyramid.iter().enumerate() {
            let mut jpeg_bytes = Vec::new();
            let mut encoder = JpegEncoder::new(&mut jpeg_bytes);
            encoder.encode_image(*image_l).unwrap();
            self.log(
                format!("{entity_path}/level{l}"), 
                &rr::EncodedImage::from_file_contents(jpeg_bytes).with_draw_order(l as f32)
            ).unwrap();
        }
    }

    fn log_features(&self, features: &[[f32; 2]], entity_path: &str) {
        self.log(
            entity_path, 
            &rr::Points2D::new(features.iter().map(|[x,y]| [x+0.5, y+0.5]))
                .with_draw_order(100.0)
                .with_colors([rr::Color::from_rgb(255, 0, 0)])
                .with_radii([1.5])
        ).unwrap();
    }

    fn log_map(&self, img: &[f32], width: u32, height: u32, cmap: Option<rr::components::Colormap>, entity_path: &str) {
        let h = height as usize;
        let w = width as usize;
        let img = rr::external::ndarray::Array2::from_shape_fn((h, w), |(i, j)| img[i*w+j]);
        self.log(
            entity_path,
            &rr::DepthImage::try_from(img).unwrap()
                .with_colormap(cmap.unwrap_or(rr::components::Colormap::Viridis))
        ).unwrap();
    }

    fn log_image_with_features_colored(&mut self, image: &[u8], width: u32, height: u32, features: &[(usize, [f32; 2])], entity_path: &str) {
        todo!()
    }

    fn set_frame(&self, frame_id: i64) {
        self.set_time_sequence("frame_id", frame_id);
    }
}
