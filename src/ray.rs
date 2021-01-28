use std::f32::consts::PI;
use std::ops::Mul;

use cgmath::{InnerSpace, Matrix3, Matrix4, Point3, Vector3, Zero};
use rand::{Rng, RngCore};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub inv_direction: Vector3<f32>
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Ray {
        let inv_direction = 1.0 / direction;
        Ray { origin, direction, inv_direction }
    }

    pub fn between(from: Point3<f32>, to: Point3<f32>) -> Ray {
        assert_ne!(from, to);

        Ray::new(from, (to - from).normalize())
    }

    pub fn reflect(incident: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
        incident - 2.0 * normal.dot(incident) * normal
    }

    pub fn refract(incident: Vector3<f32>, normal: Vector3<f32>, eta: f32) -> Vector3<f32> {
        let k = 1.0 - eta * eta * (1.0 - normal.dot(incident) * normal.dot(incident));

        if k < 0.0 {
            Vector3::zero()
        } else {
            eta * incident - (eta * normal.dot(incident) + k.sqrt()) * normal
        }
    }

    pub fn add_bias(pos: Point3<f32>, normal: Vector3<f32>, bias: f32) -> Point3<f32> {
        pos + normal * bias
    }

    pub fn perturb(&self, rng: &mut impl RngCore, magnitude: f32) -> Ray {
        let perturb = magnitude * loop {
            let perturb = Vector3 {
                x: rng.gen_range(-1.0_f32..=1.0_f32),
                y: rng.gen_range(-1.0_f32..=1.0_f32),
                z: rng.gen_range(-1.0_f32..=1.0_f32)
            };

            if perturb.magnitude2() <= 1.0 {
                break perturb;
            };
        };

        let perturb = if perturb.dot(self.direction) < 0.0 {
            -1.0 * perturb
        } else {
            perturb
        };

        Ray::new(self.origin, (self.direction + perturb).normalize())
    }

    #[inline(always)]
    pub fn transform(&self, m: &Matrix4<f32>) -> (Ray, f32) {
        let origin = Point3::from_homogeneous(m * self.origin.to_homogeneous());
        let direction = Matrix3::from_cols(m.x.truncate(), m.y.truncate(), m.z.truncate()) * self.direction;
        let dist_mult = 1.0 / direction.magnitude();

        (Ray::new(origin, direction * dist_mult), dist_mult)
    }
}

impl <'a, 'b> Mul<&'a Ray> for &'b Matrix4<f32> {
    type Output = Ray;

    fn mul(self, other: &'a Ray) -> Ray {
        other.transform(self).0
    }
}

impl <'a> Mul<&'a Ray> for Matrix4<f32> {
    type Output = Ray;

    fn mul(self, other: &'a Ray) -> Ray {
        other.transform(&self).0
    }
}

impl <'a> Mul<Ray> for &'a Matrix4<f32> {
    type Output = Ray;

    fn mul(self, other: Ray) -> Ray {
        other.transform(self).0
    }
}

impl Mul<Ray> for Matrix4<f32> {
    type Output = Ray;

    fn mul(self, other: Ray) -> Ray {
        other.transform(&self).0
    }
}
