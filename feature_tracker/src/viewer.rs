use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, PixelWithColorType, Primitive};
use image::codecs::jpeg::JpegEncoder;
use rerun as rr;

/// Viewer trait for basic visualization operations
pub trait FeatureTrackerViewer {
    
    /// Log raw image
    fn log_image_raw(&self, image: &DynamicImage, entity_path: &str);
    
    /// Log image with features drawn on it
    fn log_image_with_features(&mut self, image: &[u8], width: u32, height: u32, features: &[[f32; 2]], entity_path: &str);

    fn log_image_pyramid(&self, pyramid: &[&DynamicImage], entity_path: &str);
    // where 
    //     Luma<T>: PixelWithColorType,
    //     ImageBuffer<Luma<T>, Vec<T>>: GenericImageView;
    
    /// Log image with features colored by feature ID
    /// Features should be provided as (feature_id, [x, y]) tuples
    fn log_image_with_features_colored(&mut self, image: &[u8], width: u32, height: u32, features: &[(usize, [f32; 2])], entity_path: &str);
    
    /// Set the current frame/timestamp
    fn set_frame(&self, frame_id: i64);
    
}


impl FeatureTrackerViewer for rr::RecordingStream {
    fn log_image_raw(&self, in_image: &DynamicImage, entity_path: &str) {
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

    fn log_image_with_features(&mut self, image: &[u8], width: u32, height: u32, features: &[[f32; 2]], entity_path: &str) {
        todo!()
    }

    fn log_image_with_features_colored(&mut self, image: &[u8], width: u32, height: u32, features: &[(usize, [f32; 2])], entity_path: &str) {
        todo!()
    }

    fn set_frame(&self, frame_id: i64) {
        self.set_time_sequence("frame_id", frame_id);
    }
}
