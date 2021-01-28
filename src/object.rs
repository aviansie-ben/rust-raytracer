use std::f32;
use std::ops::Mul;
use std::sync::Arc;

use cgmath::{EuclideanSpace, InnerSpace, Matrix3, Matrix4, Point2, Point3, Rad, SquareMatrix, Vector3};
use cgmath::num_traits::Float;

use crate::material::{Material, PointMaterial};
use crate::mesh::TriMesh;
use crate::ray::Ray;

pub trait Intersectible<'a> {
    type Intersection;

    fn find_intersection(&'a self, ray: &Ray) -> Option<Self::Intersection>;
    fn intersects(&'a self, ray: &Ray) -> bool {
        self.find_intersection(ray).is_some()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>
}

impl BoundingBox {
    pub fn empty() -> BoundingBox {
        BoundingBox {
            min: Point3::origin(),
            max: Point3::origin()
        }
    }

    pub fn of_vertices<'a>(vertices: impl IntoIterator<Item=&'a Point3<f32>>) -> BoundingBox {
        let mut vertices = vertices.into_iter();

        let (mut min, mut max) = match vertices.next() {
            Some(first_vertex) => (*first_vertex, *first_vertex),
            None => return BoundingBox::empty()
        };

        for vertex in vertices {
            if vertex.x < min.x { min.x = vertex.x; };
            if vertex.x > max.x { max.x = vertex.x; };

            if vertex.y < min.y { min.y = vertex.y; };
            if vertex.y > max.y { max.y = vertex.y; };

            if vertex.z < min.z { min.z = vertex.z; };
            if vertex.z > max.z { max.z = vertex.z; };
        }

        BoundingBox { min, max }
    }

    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    pub fn center(&self) -> Point3<f32> {
        Point3 {
            x: 0.5 * (self.max.x + self.min.x),
            y: 0.5 * (self.max.y + self.min.y),
            z: 0.5 * (self.max.z + self.min.z)
        }
    }

    pub fn corners(&self) -> [Point3<f32>; 8] {
        [
            self.min,
            Point3 { x: self.max.x, y: self.min.y, z: self.min.z },
            Point3 { x: self.min.x, y: self.max.y, z: self.min.z },
            Point3 { x: self.max.x, y: self.max.y, z: self.min.z },
            Point3 { x: self.min.x, y: self.min.y, z: self.max.z },
            Point3 { x: self.max.x, y: self.min.y, z: self.max.z },
            Point3 { x: self.min.x, y: self.max.y, z: self.max.z },
            self.max
        ]
    }

    pub fn combine_with(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min: Point3 {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y),
                z: self.min.z.min(other.min.z)
            },
            max: Point3 {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y),
                z: self.max.z.max(other.max.z)
            }
        }
    }

    pub fn transform(&self, tf: &Matrix4<f32>) -> BoundingBox {
        let mut min = Point3 { x: f32::infinity(), y: f32::infinity(), z: f32::infinity() };
        let mut max = Point3 { x: f32::neg_infinity(), y: f32::neg_infinity(), z: f32::neg_infinity() };

        for corner in self.corners().iter() {
            let corner = Point3::from_homogeneous(tf * corner.to_homogeneous());

            if corner.x < min.x { min.x = corner.x };
            if corner.x > max.x { max.x = corner.x };

            if corner.y < min.y { min.y = corner.y };
            if corner.y > max.y { max.y = corner.y };

            if corner.z < min.z { min.z = corner.z };
            if corner.z > max.z { max.z = corner.z };
        };

        BoundingBox { min, max }
    }
}

impl <'a> Intersectible<'a> for BoundingBox {
    type Intersection = f32;

    fn find_intersection(&self, ray: &Ray) -> Option<f32> {
        fn sort2(a: f32, b: f32) -> (f32, f32) {
            if a < b {
                (a, b)
            } else {
                (b, a)
            }
        }

        fn find_axis_intersection(min: f32, max: f32, origin: f32, inv_dir: f32) -> (f32, f32) {
            sort2(
                (min - origin) * inv_dir,
                (max - origin) * inv_dir
            )
        }

        let (txmin, txmax) = find_axis_intersection(self.min.x, self.max.x, ray.origin.x, ray.inv_direction.x);
        let (tymin, tymax) = find_axis_intersection(self.min.y, self.max.y, ray.origin.y, ray.inv_direction.y);
        let (tzmin, tzmax) = find_axis_intersection(self.min.z, self.max.z, ray.origin.z, ray.inv_direction.z);

        let tmin = txmin.max(tymin).max(tzmin);
        let tmax = txmax.min(tymax).min(tzmax);

        if tmax >= tmin && tmax >= 0.0 {
            Some(tmin)
        } else {
            None
        }
    }
}

impl <'a, 'b> Mul<&'a BoundingBox> for &'b Matrix4<f32> {
    type Output = BoundingBox;

    fn mul(self, other: &'a BoundingBox) -> BoundingBox {
        other.transform(self)
    }
}

impl <'a> Mul<&'a BoundingBox> for Matrix4<f32> {
    type Output = BoundingBox;

    fn mul(self, other: &'a BoundingBox) -> BoundingBox {
        other.transform(&self)
    }
}

impl <'a> Mul<BoundingBox> for &'a Matrix4<f32> {
    type Output = BoundingBox;

    fn mul(self, other: BoundingBox) -> BoundingBox {
        other.transform(self)
    }
}

impl Mul<BoundingBox> for Matrix4<f32> {
    type Output = BoundingBox;

    fn mul(self, other: BoundingBox) -> BoundingBox {
        other.transform(&self)
    }
}

pub struct Intersection<'a> {
    pub pos: Point3<f32>,
    pub normal: Vector3<f32>,
    pub tex_coord: Point2<f32>,

    pub distance: f32,

    pub material: &'a Material
}

impl <'a> Intersection<'a> {
    pub fn point_material(&self) -> PointMaterial {
        self.material.at_point(self.tex_coord)
    }

    pub fn transform(&mut self, transform: &Matrix4<f32>, dist_mult: f32) -> () {
        self.pos = Point3::from_homogeneous(transform * self.pos.to_homogeneous());
        self.normal = (transform * self.normal.extend(0.0)).truncate().normalize();
        self.distance *= dist_mult;
    }
}

#[derive(Debug, Clone)]
pub enum ObjectKind {
    TriMesh(Arc<TriMesh>),
    Sphere
}

#[derive(Debug, Clone)]
pub struct Object {
    transform: Matrix4<f32>,
    inv_transform: Matrix4<f32>,

    material: Arc<Material>,

    aabb: BoundingBox,
    obb: BoundingBox,

    kind: ObjectKind
}

impl Object {
    pub fn new(
        transform: Matrix4<f32>,
        material: Arc<Material>,
        obb: BoundingBox,
        kind: ObjectKind
    ) -> Object {
        let inv_transform = transform.invert().unwrap();
        let aabb = transform * obb;

        Object {
            transform,
            inv_transform,
            material,
            aabb,
            obb,
            kind
        }
    }

    pub fn new_sphere(
        center: Point3<f32>,
        radius: f32,
        material: Arc<Material>
    ) -> Object {
        let transform = Matrix4::from_translation(center.to_vec()) * Matrix4::from_scale(radius);
        let obb = BoundingBox {
            min: Point3 { x: -1.0, y: -1.0, z: -1.0 },
            max: Point3 { x: 1.0, y: 1.0, z: 1.0 }
        };

        Object::new(transform, material, obb, ObjectKind::Sphere)
    }

    pub fn new_mesh(
        mesh: Arc<TriMesh>,
        pos: Point3<f32>,
        rot: Vector3<f32>,
        scale: f32,
        material: Arc<Material>
    ) -> Object {
        let transform = Matrix4::from_translation(pos.to_vec()) * orientation_matrix(rot) * Matrix4::from_scale(scale);

        Object::new(transform, material, *mesh.bounding_box(), ObjectKind::TriMesh(mesh))
    }

    pub fn transform(&self) -> &Matrix4<f32> {
        &self.transform
    }

    pub fn inv_transform(&self) -> &Matrix4<f32> {
        &self.inv_transform
    }

    pub fn aabb(&self) -> &BoundingBox {
        &self.aabb
    }
}

fn find_sphere_intersection<'a>(o: &'a Object, r: &Ray) -> Option<Intersection<'a>> {
    let origin = r.origin.to_vec();
    let b = (2.0 * r.direction).dot(origin);
    let c = origin.dot(origin) - 1.0;

    let qterm = b * b - 4.0 * c;

    if qterm < 0.0 {
        return None;
    };

    let sqterm = qterm.sqrt();

    let mut t = -(b + sqterm) * 0.5;

    if t < 0.0 {
        t = -(b - sqterm) * 0.5;

        if t < 0.0 {
            return None;
        };
    };

    let pos = r.origin + r.direction * t;

    Some(Intersection {
        pos,
        normal: pos.to_vec(),
        tex_coord: Point2 {
            x: 0.5 + pos.z.atan2(pos.x) * 1.0 / (2.0 * f32::consts::PI),
            y: 0.5 + pos.y.asin() * 1.0 / (2.0 * f32::consts::PI)
        },

        distance: t,

        material: &o.material
    })
}

impl <'a> Intersectible<'a> for Object {
    type Intersection = Intersection<'a>;

    fn find_intersection(&'a self, r: &Ray) -> Option<Intersection<'a>> {
        match self.kind {
            ObjectKind::Sphere => find_sphere_intersection(self, r),
            ObjectKind::TriMesh(ref mesh) => mesh.find_intersection(r).map(|(v, dist)| {
                Intersection {
                    pos: v.pos,
                    normal: v.normal,
                    tex_coord: v.tex_coord,

                    distance: dist,

                    material: &self.material
                }
            })
        }
    }
}

pub fn orientation_matrix(rot: Vector3<f32>) -> Matrix4<f32> {
    let x = Matrix4::from_angle_y(Rad(rot.x));
    let y = Matrix4::from_angle_x(Rad(rot.y));
    let z = Matrix4::from_angle_z(Rad(rot.z));

    // TODO Is this correct?
    x * y * z
}
