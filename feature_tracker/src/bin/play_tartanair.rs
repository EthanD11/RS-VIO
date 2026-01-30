use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use feature_tracker::players::tartanair_player::{self, *};
use feature_tracker::feature_tracker::*;


fn main() -> Result<(), Box<dyn Error>> {

    let config = fs::read_to_string("config/config.yaml")?;
    let config: FeatureTrackingConfig = serde_yaml::from_str(&config)?;
    
    
    let dataset_path = Path::new("/home/ethan/Documents/datasets/tartanair/neighborhood/P001");
    let player = TartanAirPlayer::new(dataset_path);
    player.run(config)?;
    // tracker.process_frame(image);

    Ok(())
}