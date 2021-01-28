use cgmath::{ElementWise, EuclideanSpace, Point2, Vector3};
use image::{Luma, Rgb};

use texture::Texture2D;

#[derive(Debug, Clone)]
pub struct Material {
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,

    pub opacity: f32,

    pub reflectance: f32,
    pub reflection_gloss: f32,
    pub transmittance: f32,
    pub refractive_index: f32,
    pub refraction_gloss: f32,

    pub diffuse_map: Option<Texture2D<Rgb<u8>>>,
    pub ao_map: Option<Texture2D<Luma<u8>>>
}

impl Material {
    pub fn new(
        ambient: Vector3<f32>,
        diffuse: Vector3<f32>,
        specular: Vector3<f32>,
        shininess: f32
    ) -> Material {
        Material {
            ambient,
            diffuse,
            specular,
            shininess,

            opacity: 1.0,

            reflectance: 0.0,
            reflection_gloss: 0.05,
            transmittance: 0.0,
            refractive_index: 1.0,
            refraction_gloss: 0.0,

            diffuse_map: None,
            ao_map: None
        }
    }

    pub fn reflective(mut self, reflectance: f32) -> Material {
        self.reflectance = reflectance;
        self
    }

    pub fn translucent(mut self, opacity: f32, transmittance: f32, refractive_index: f32) -> Material {
        self.opacity = opacity;
        self.transmittance = transmittance;
        self.refractive_index = refractive_index;
        self
    }

    pub fn at_point(&self, tex_coord: Point2<f32>) -> PointMaterial {
        let mut mtl = PointMaterial {
            ambient: self.ambient,
            diffuse: self.diffuse,
            specular: self.specular,
            shininess: self.shininess,

            opacity: self.opacity,

            reflectance: self.reflectance,
            reflection_gloss: self.reflection_gloss,
            transmittance: self.transmittance,
            refractive_index: self.refractive_index,
            refraction_gloss: self.refraction_gloss
        };

        if let Some(diffuse_map) = self.diffuse_map.as_ref() {
            let d = diffuse_map.get(tex_coord - Point2::origin());
            mtl.ambient.mul_assign_element_wise(d);
            mtl.diffuse.mul_assign_element_wise(d);
        };

        if let Some(ao_map) = self.ao_map.as_ref() {
            mtl.ambient.mul_assign_element_wise(ao_map.get(tex_coord - Point2::origin()));
        };

        mtl
    }
}

#[derive(Debug, Clone)]
pub struct PointMaterial {
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,

    pub opacity: f32,

    pub reflectance: f32,
    pub reflection_gloss: f32,
    pub transmittance: f32,
    pub refractive_index: f32,
    pub refraction_gloss: f32
}
