use std::ops::Mul;

use cgmath::{ElementWise, EuclideanSpace, InnerSpace, Point3, Vector3, Zero};

use crate::material::PointMaterial;
use crate::object::Intersection;
use crate::ray::Ray;

pub struct PointLight {
    pub pos: Point3<f32>,
    pub radius: f32,

    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,

    pub attenuation: Vector3<f32>
}

impl PointLight {
    pub fn new() -> PointLight {
        PointLight {
            pos: Point3::origin(),
            radius: 0.0,
            ambient: Vector3::zero(),
            diffuse: Vector3::zero(),
            specular: Vector3::zero(),
            attenuation: Vector3::zero()
        }
    }
    pub fn calculate_attenuation(&self, dist: f32) -> f32 {
        1.0 / (self.attenuation.x + self.attenuation.y * dist + self.attenuation.z * dist * dist)
    }

    pub fn calculate_attenuation_for(&self, pos: Point3<f32>) -> f32 {
        self.calculate_attenuation((pos - self.pos).magnitude())
    }

    pub fn calculate_illumination(
        &self,
        ray: &Ray,
        i: &Intersection,
        mat: &PointMaterial,
        visibility: f32
    ) -> Vector3<f32> {
        let mut result = self.ambient.mul_element_wise(mat.ambient);

        if visibility > 0.0 {
            let obj_to_light = (self.pos - i.pos).normalize();

            result += (i.normal.dot(obj_to_light).max(0.0) * visibility)
                .mul(self.diffuse)
                .mul_element_wise(mat.diffuse);

            result += (-ray.direction.dot(Ray::reflect(-obj_to_light, i.normal))).max(0.0)
                .powf(mat.shininess)
                .mul(self.specular)
                .mul_element_wise(mat.specular);
        };

        result * self.calculate_attenuation_for(i.pos)
    }
}
