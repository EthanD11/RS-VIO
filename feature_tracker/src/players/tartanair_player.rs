use std::path::{Path, PathBuf};
use std::fs;
use std::thread;
use std::time;

use anyhow::Result;
use image::{self, ImageBuffer, Luma};
use rerun as rr;

use crate::feature_tracker::*;


#[derive(Clone)]
pub struct TartanAirPlayer {
    dataset_path: PathBuf
}

impl TartanAirPlayer {
    pub fn new<P: AsRef<Path>>(dataset_path: P) -> Self {
        TartanAirPlayer { dataset_path: dataset_path.as_ref().to_path_buf() }
    }

    pub fn run(&self, config: FeatureTrackingConfig) -> Result<()> {
        let mut tracker = FeatureTracker::new(config);
        
        let rec = rr::RecordingStreamBuilder::new("Patch Tracker").spawn()?;
        let left_entity_path = "image_left";

        let mut left_images: Vec<_> = fs::read_dir(&(self.dataset_path.join("image_left")))?
            .map(|r| r.unwrap())
            .collect();
        left_images.sort_by_key(|dir_entry| dir_entry.path());
        
        for file in left_images.iter().skip(300).take(100) {
            let left_image = image::open(file.path())?;

            tracker.process_frame(&left_image.to_luma8());
            rec.log(
                left_entity_path,
                &rr::Image::from_dynamic_image(left_image)?
            )?;
        }
        


        Ok(())
    }
    
}