use crate::feature_tracker::Feature;

pub struct Frame {
    pub frame_id: i64,
    pub features: Vec<Feature>
}

impl Frame {
    pub fn new(id: i64) -> Frame {
        Frame { 
            frame_id: id, 
            features: Vec::new() 
        }
    }
}