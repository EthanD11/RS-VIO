use std::path::{Path, PathBuf};
use std::fs;
use std::thread;
use std::time;

use anyhow::Result;
use image::{self};
use rerun as rr;

use crate::feature_tracker::*;
use crate::ext::Frame;
use crate::viewer::FeatureTrackerViewer;


#[derive(Clone)]
pub struct TartanAirPlayer {
    dataset_path: PathBuf
}

impl TartanAirPlayer {
    pub fn new<P: AsRef<Path>>(dataset_path: P) -> Self {
        TartanAirPlayer { dataset_path: dataset_path.as_ref().to_path_buf() }
    }

    pub fn run(&self, config: FeatureTrackingConfig) -> Result<()> {
        let rec = rr::RecordingStreamBuilder::new("Patch Tracker").spawn()?;

        let mut tracker = FeatureTracker::new(config, Some(&rec as &dyn FeatureTrackerViewer));
        // let mut tracker = FeatureTracker::new(config, None);
        

        let mut left_images: Vec<_> = fs::read_dir(&(self.dataset_path.join("image_left")))?
            .map(|r| r.unwrap())
            .collect();
        left_images.sort_by_key(|dir_entry| dir_entry.path());
        
        for file in left_images.iter().skip(300).take(10) {

            let frame_id = file.path()
                .file_stem().unwrap()
                .to_str().unwrap()
                .split("_").nth(0).unwrap()
                .parse().unwrap();

            rec.set_time_sequence("frame_id", frame_id);

            let frame_image = image::open(file.path())?;
            let frame = Frame::new(frame_id);

            tracker.process_frame(&frame_image.to_luma32f(), frame);



            rec.log_image_raw(&frame_image, "image/colored");
        }
        


        Ok(())
    }
    
}
