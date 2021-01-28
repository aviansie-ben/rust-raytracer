use std::path::Path;

use cgmath::{Vector2, Vector3, Zero};
use image::{ImageBuffer, ImageError, ImageResult, Luma, Pixel, Rgb};
use image::error::{LimitError, LimitErrorKind};

fn interpolate_color(a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>, d: Vector3<f32>, x: f32, y: f32) -> Vector3<f32> {
    debug_assert!(x >= 0.0 && x <= 1.0);
    debug_assert!(y >= 0.0 && y <= 1.0);

    let ab = (1.0 - x) * a + x * b;
    let cd = (1.0 - x) * c + x * d;

    (1.0 - y) * ab + y * cd
}

#[derive(Debug, Clone)]
pub struct Texture2D<P: Pixel<Subpixel=u8> + 'static> {
    width: u32,
    height: u32,
    size_f: Vector2<f32>,
    buf: Vec<u8>,
    _phantom: std::marker::PhantomData<P>
}

impl Texture2D<Luma<u8>> {
    pub fn load_luma(path: &Path) -> ImageResult<Self> {
        Texture2D::new(image::open(path)?.to_luma())
    }
}

impl Texture2D<Rgb<u8>> {
    pub fn load_rgb(path: &Path) -> ImageResult<Self> {
        Texture2D::new(image::open(path)?.to_rgb())
    }
}

impl <P: Pixel<Subpixel=u8> + 'static> Texture2D<P> {
    pub fn new(img: ImageBuffer<P, Vec<u8>>) -> ImageResult<Self> {
        let width = img.width();
        let height = img.height();

        if width == 0 || width > 16777216 || height == 0 || height > 16777216 {
            return Result::Err(ImageError::Limits(LimitError::from_kind(LimitErrorKind::DimensionError)));
        };

        let mut buf = img.into_raw();

        for val in buf.iter_mut() {
            *val = ((*val as f32 / 255.0).powf(::render::INV_GAMMA) * 255.0) as u8;
        };

        Result::Ok(Texture2D {
            width,
            height,
            size_f: Vector2 { x: (width - 1) as f32, y: (height - 1) as f32 },
            buf,
            _phantom: std::marker::PhantomData
        })
    }

    unsafe fn get_pixel_unchecked(&self, tc: Vector2<usize>) -> Vector3<f32> {
        debug_assert!(tc.x < self.width as usize);
        debug_assert!(tc.y < self.height as usize);

        let no_channels = <P as Pixel>::channel_count() as usize;
        let index = no_channels * (tc.y * self.width as usize + tc.x);

        let ptr = self.buf.get_unchecked(index) as *const P::Subpixel;
        let pix = <P as Pixel>::from_slice(std::slice::from_raw_parts(ptr, no_channels)).to_rgb();

        let byte_to_float = (255.0 as f32).recip();

        return Vector3 {
            x: pix.0[0] as f32 * byte_to_float,
            y: pix.0[1] as f32 * byte_to_float,
            z: pix.0[2] as f32 * byte_to_float
        }
    }

    unsafe fn get_pixel_unchecked_f(&self, tc: Vector2<f32>) -> Vector3<f32> {
        self.get_pixel_unchecked(Vector2 { x: tc.x as usize, y: tc.y as usize })
    }

    pub fn get(&self, tc: Vector2<f32>) -> Vector3<f32> {
        if tc.x < 0.0 || tc.x > 1.0 || tc.y < 0.0 || tc.y > 1.0 {
            Vector3::zero()
        } else {
            let tc = Vector2 { x: tc.x * self.size_f.x, y: tc.y * self.size_f.y };
            let x_floor = tc.x.floor();
            let x_ceil = tc.x.ceil();
            let y_floor = tc.y.floor();
            let y_ceil = tc.y.ceil();

            // This is safe because we know the integer coordinates must be within the bounds of the
            // image, since they were originally in the bounds [0.0, 1.0] and were multiplied by the
            // image size. The image size is exactly representable as f32s due to the check in
            // Texture2D::new.
            let (a, b, c, d) = unsafe {
                let a = self.get_pixel_unchecked_f(Vector2 { x: x_floor, y: y_floor });
                let b = self.get_pixel_unchecked_f(Vector2 { x: x_ceil, y: y_floor });
                let c = self.get_pixel_unchecked_f(Vector2 { x: x_floor, y: y_ceil });
                let d = self.get_pixel_unchecked_f(Vector2 { x: x_ceil, y: y_ceil });

                (a, b, c, d)
            };

            interpolate_color(a, b, c, d, tc.x - x_floor, tc.y - y_floor)
        }
    }
}
