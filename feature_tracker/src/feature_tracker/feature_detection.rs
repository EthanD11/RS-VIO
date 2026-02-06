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
    let corners = suppress_non_maximum(&score_map, 1, threshold, viewer);
    let corners = suppress::local_maxima(&corners, min_dist_between_points);
    
    let (w, h) = image_fine.dimensions();

    let buffer = u32::max(min_dist_between_points, 50);
    let xrange = min_dist_between_points..(w-min_dist_between_points);
    let yrange = min_dist_between_points..(h-min_dist_between_points);
    corners
        .into_iter()
        .filter(|c| c.from_past_feature.not())
        .filter(|c| xrange.contains(&c.x) && yrange.contains(&c.y))
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

    if log::max_level() >= log::LevelFilter::Debug {
        if let Some(v) = viewer {
            v.log_map(dimage_dxx.as_raw(), w, h, "image/shi_tomasi/grad/x", None, None);
            v.log_map(dimage_dyy.as_raw(), w, h, "image/shi_tomasi/grad/y", None, None);
        }
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

    if log::max_level() >= log::LevelFilter::Debug {
        if let Some(v) = viewer {
            v.log_map(dimage_dxx.as_raw(), w, h, "image/shi_tomasi/structure/dxx", None, None);
            v.log_map(dimage_dyy.as_raw(), w, h, "image/shi_tomasi/structure/dyy", None, None);
            v.log_map(dimage_dxy.as_raw(), w, h, "image/shi_tomasi/structure/dxy", None, None);
        }
    }

    let dimage_dxx = fast_blur(&dimage_dxx, detection_blur);
    let dimage_dyy = fast_blur(&dimage_dyy, detection_blur);
    let dimage_dxy = fast_blur(&dimage_dxy, detection_blur);

    if log::max_level() >= log::LevelFilter::Debug {
        if let Some(v) = viewer {
            v.log_map(dimage_dxx.as_raw(), w, h, "image/shi_tomasi/structure_smooth/dxx", None, None);
            v.log_map(dimage_dyy.as_raw(), w, h, "image/shi_tomasi/structure_smooth/dyy", None, None);
            v.log_map(dimage_dxy.as_raw(), w, h, "image/shi_tomasi/structure_smooth/dxy", None, None);
        }
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

            unsafe {
                score_map.unsafe_put_pixel(x, y, Luma::<f32>([score]));
            }
        }
    }

    if log::max_level() >= log::LevelFilter::Info {
        if let Some(v) = viewer {
            v.log_map(&score_map, w, h, "image/shi_tomasi/score_map", None, None);
        }
    }
    
    score_map
}


/// Code (lightly) adapte form imageproc::suppress::suppress_non_maximum
/// Returned image has zeroes for all inputs pixels which do not have the greatest
/// intensity in the (2 * radius + 1) square block centred on them.
/// Ties are resolved lexicographically.
pub fn suppress_non_maximum<I>(score_map: &I, radius: u32, threshold: Float, viewer: Option<&dyn FeatureTrackerViewer>) -> Vec<ShiTomasiCorner>
where
    I: GenericImage<Pixel = Luma<Float>>
{
    let (width, height) = score_map.dimensions();
    let mut out = Vec::new();
    if width == 0 || height == 0 {
        return out;
    }

    let mut suppressed_score_map = if viewer.is_some() && log::max_level() >= log::LevelFilter::Debug  {
        FloatGrayImage::new(width, height)
    } else {
        FloatGrayImage::new(0, 0)

    };

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
            let mut best_score = unsafe { score_map.unsafe_get_pixel(x, y)[0] };

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
                    out.push(ShiTomasiCorner::new(best_x, best_y, best_score));

                    if viewer.is_some() && log::max_level() >= log::LevelFilter::Debug {
                        suppressed_score_map.put_pixel(best_x, best_y, Luma([best_score]));
                    } 
                }
            }

        }
    }

    if let Some(v) = viewer && log::max_level() >= log::LevelFilter::Debug {
        v.log_map(&suppressed_score_map, width, height, "image/shi_tomasi/score_map_suppressed", Some(rerun::components::Colormap::Turbo), Some(-25.0));
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