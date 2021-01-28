use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector2, Vector3};
use cgmath::num_traits::Float;
use itertools::Itertools;

use crate::bvh::Bvh;
use crate::object::{BoundingBox, Intersectible};
use crate::ray::Ray;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VertexId(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tri<T>(pub T, pub T, pub T);

impl <T> Tri<T> {
    pub fn map<U>(self, mut f: impl FnMut (T) -> U) -> Tri<U> {
        Tri(f(self.0), f(self.1), f(self.2))
    }

    pub fn map_result<U, V>(self, mut f: impl FnMut (T) -> Result<U, V>) -> Result<Tri<U>, V> {
        Result::Ok(Tri(f(self.0)?, f(self.1)?, f(self.2)?))
    }
}

impl Tri<VertexId> {
    pub unsafe fn deref<'a>(&self, vertices: &'a [Vertex]) -> Tri<&'a Vertex> {
        Tri(
            vertices.get_unchecked((self.0).0 as usize),
            vertices.get_unchecked((self.1).0 as usize),
            vertices.get_unchecked((self.2).0 as usize)
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    pub pos: Point3<f32>,
    pub normal: Vector3<f32>,
    pub tex_coord: Point2<f32>
}

#[derive(Debug)]
pub struct TriMesh {
    vertices: Vec<Vertex>,
    triangles: Bvh<Tri<VertexId>>
}

impl TriMesh {
    pub unsafe fn from_parts_unchecked(vertices: Vec<Vertex>, triangles: Bvh<Tri<VertexId>>) -> TriMesh {
        TriMesh { vertices, triangles }
    }

    pub fn from_parts(vertices: Vec<Vertex>, triangles: Bvh<Tri<VertexId>>) -> TriMesh {
        assert!(vertices.len() < u32::max_value() as usize);

        for t in triangles.iter() {
            assert!(((t.0).0 as usize) < vertices.len());
            assert!(((t.1).0 as usize) < vertices.len());
            assert!(((t.2).0 as usize) < vertices.len());
        };

        unsafe { TriMesh::from_parts_unchecked(vertices, triangles) }
    }

    pub fn bounding_box(&self) -> &BoundingBox {
        self.triangles.bounding_box()
    }
}

impl <'a> Intersectible<'a> for TriMesh {
    type Intersection = (Vertex, f32);

    fn find_intersection(&self, ray: &Ray) -> Option<(Vertex, f32)> {
        let mut depth = f32::infinity();
        let mut intersection = None;

        self.triangles.search(ray, |tri| {
            // We don't want to incur the cost of performing a runtime check to make sure that all
            // vertex indices are in range on every intersection test. Instead, we make sure that
            // all safe ways of creating or modifying a TriMesh will always ensure that no triangles
            // in the TriMesh have out-of-bounds indices. Because of this, it's perfectly fine to
            // skip the test here and just use the indices directly.
            let Tri(a, b, c) = unsafe { tri.deref(&self.vertices) };

            // Use the Möller-Trumbore algorithm for intersection
            let ab = b.pos - a.pos;
            let ac = c.pos - a.pos;

            let pvec = ray.direction.cross(ac);
            let det = ab.dot(pvec);

            // If det is close to zero, the ray is parallel to the triangle and we can bail early
            if det.abs() <= 1e-7 {
                return;
            };

            let inv_det = 1.0 / det;

            let tvec = ray.origin - a.pos;
            let u = tvec.dot(pvec) * inv_det;

            if u < 0.0 || u > 1.0 {
                return;
            };

            let qvec = tvec.cross(ab);
            let v = ray.direction.dot(qvec) * inv_det;

            if v < 0.0 || u + v > 1.0 {
                return;
            };

            let t = ac.dot(qvec) * inv_det;

            if t < 0.0 || t >= depth {
                return;
            };

            depth = t;
            intersection = Some((a, b, c, u, v));
        });

        if let Some((a, b, c, u, v)) = intersection {
            // Use Barycentric interpolation to find the intersection parameters
            Some((Vertex {
                pos: ray.origin + depth * ray.direction,
                normal: ((1.0 - u - v) * a.normal + u * b.normal + v * c.normal).normalize(),
                tex_coord: Point2::from_vec(
                    (1.0 - u - v) * a.tex_coord.to_vec() + u * b.tex_coord.to_vec() + v * c.tex_coord.to_vec()
                )
            }, depth))
        } else {
            None
        }
    }
}

pub struct TriMeshBuilder {
    vertices: Vec<Vertex>,
    triangles: Vec<Tri<VertexId>>
}

impl TriMeshBuilder {
    pub fn new() -> TriMeshBuilder {
        TriMeshBuilder { vertices: vec![], triangles: vec![] }
    }

    pub fn add_vertex(&mut self, v: Vertex) -> Result<VertexId, ()> {
        if self.vertices.len() < u32::max_value() as usize {
            self.vertices.push(v);
            Result::Ok(VertexId(self.vertices.len() as u32 - 1))
        } else {
            Result::Err(())
        }
    }

    pub unsafe fn add_triangle_unchecked(&mut self, t: Tri<VertexId>) -> () {
        self.triangles.push(t);
    }

    pub fn add_triangle(&mut self, t: Tri<VertexId>) -> () {
        assert!(((t.0).0 as usize) < self.vertices.len());
        assert!(((t.1).0 as usize) < self.vertices.len());
        assert!(((t.2).0 as usize) < self.vertices.len());

        unsafe {
            self.add_triangle_unchecked(t);
        };
    }

    pub fn build(self, bvh_delta: usize) -> TriMesh {
        // Since the TriMeshBuilder has been validating any added triangles to make sure that none
        // contain out-of-bounds indices, we can be sure that no such triangles could have been
        // added safely. As a result, we don't need to do any bounds checking here.
        let vertices = self.vertices;
        let triangles = Bvh::new(self.triangles, |tri| {
            let tri = unsafe { tri.deref(&vertices) };

            BoundingBox {
                min: Point3 {
                    x: tri.0.pos.x.min(tri.1.pos.x).min(tri.2.pos.x),
                    y: tri.0.pos.y.min(tri.1.pos.y).min(tri.2.pos.y),
                    z: tri.0.pos.z.min(tri.1.pos.z).min(tri.2.pos.z)
                },
                max: Point3 {
                    x: tri.0.pos.x.max(tri.1.pos.x).max(tri.2.pos.x),
                    y: tri.0.pos.y.max(tri.1.pos.y).max(tri.2.pos.y),
                    z: tri.0.pos.z.max(tri.1.pos.z).max(tri.2.pos.z)
                }
            }
        }, bvh_delta);

        unsafe { TriMesh::from_parts_unchecked(vertices, triangles) }
    }
}

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    SyntaxError(u64, String),
    IndexOutOfRange(u64, u32),
    TooManyVertices
}

impl From<io::Error> for LoadError {
    fn from(err: io::Error) -> LoadError {
        LoadError::Io(err)
    }
}

#[derive(Debug, Clone)]
pub enum LoadStage {
    Read,
    ConstructBvh
}

pub fn load_from_obj_file(path: &Path, bvh_delta: usize) -> Result<TriMesh, LoadError> {
    fn parse_float(part: &str, n: u64) -> Result<f32, LoadError> {
        if let Result::Ok(val) = part.parse::<f32>() {
            Result::Ok(val)
        } else {
            return Result::Err(LoadError::SyntaxError(
                n,
                format!("Cannot parse '{}' as a float", part)
            ));
        }
    }

    fn parse_u32(part: &str, n: u64) -> Result<u32, LoadError> {
        if let Result::Ok(val) = part.parse::<u32>() {
            Result::Ok(val)
        } else {
            return Result::Err(LoadError::SyntaxError(
                n,
                format!("Cannot parse '{}' as a 32-bit integer", part)
            ));
        }
    }

    fn parse_vec2(parts: &[&str], n: u64) -> Result<Vector2<f32>, LoadError> {
        assert_eq!(parts.len(), 2);
        Result::Ok(Vector2 {
            x: parse_float(parts[0], n)?,
            y: parse_float(parts[1], n)?
        })
    }

    fn parse_vec3(parts: &[&str], n: u64) -> Result<Vector3<f32>, LoadError> {
        assert_eq!(parts.len(), 3);
        Result::Ok(Vector3 {
            x: parse_float(parts[0], n)?,
            y: parse_float(parts[1], n)?,
            z: parse_float(parts[2], n)?
        })
    }

    fn parse_vertex(spec: &str, n: u64) -> Result<(u32, u32, u32), LoadError> {
        if let Some((pos, normal, tex_coord)) = spec.split('/').collect_tuple() {
            Result::Ok((
                parse_u32(pos, n)?,
                parse_u32(normal, n)?,
                parse_u32(tex_coord, n)?
            ))
        } else {
            Result::Err(LoadError::SyntaxError(n, format!("Invalid face vertex specification")))
        }
    }

    fn parse_tri(parts: &[&str], n: u64) -> Result<Tri<(u32, u32, u32)>, LoadError> {
        assert_eq!(parts.len(), 3);
        Result::Ok(Tri(
            parse_vertex(parts[0], n)?,
            parse_vertex(parts[1], n)?,
            parse_vertex(parts[2], n)?
        ))
    }

    let f = BufReader::new(File::open(path)?);

    let mut vertex_pos = vec![];
    let mut vertex_normal = vec![];
    let mut vertex_tex_coord = vec![];

    let mut vertices = HashMap::new();
    let mut builder = TriMeshBuilder::new();

    for (n, line) in (1u64..).zip(f.lines()) {
        let mut line = &(line?)[..];

        if let Some(comment_pos) = line.find('#') {
            line = &line[..comment_pos];
        };

        line = line.trim();

        if line.len() == 0 {
            continue;
        };

        let parts = line.split_whitespace().collect_vec();

        match parts[0] {
            "v" => {
                if parts.len() != 4 {
                    return Result::Err(LoadError::SyntaxError(
                        n,
                        "Wrong number of arguments to command 'v'".to_owned()
                    ));
                };

                vertex_pos.push(Point3::from_vec(parse_vec3(&parts[1..4], n)?));
            },
            "vn" => {
                if parts.len() != 4 {
                    return Result::Err(LoadError::SyntaxError(
                        n,
                        "Wrong number of arguments to command 'vn'".to_owned()
                    ));
                };

                vertex_normal.push(parse_vec3(&parts[1..4], n)?.normalize());
            },
            "vt" => {
                if parts.len() != 3 {
                    return Result::Err(LoadError::SyntaxError(
                        n,
                        "Wrong number of arguments to command 'vt'".to_owned()
                    ));
                };

                let tc = parse_vec2(&parts[1..3], n)?;

                vertex_tex_coord.push(Point2 { x: tc.x, y: 1.0 - tc.y });
            },
            "f" => {
                if parts.len() != 4 {
                    return Result::Err(LoadError::SyntaxError(
                        n,
                        "Wrong number of arguments to command 'f'".to_owned()
                    ));
                };

                let tri = parse_tri(&parts[1..4], n)?.map_result(|(pos, tex_coord, normal)| {
                    match vertices.entry((pos, tex_coord, normal)) {
                        Entry::Occupied(entry) => {
                            Result::Ok(*entry.get())
                        },
                        Entry::Vacant(entry) => {
                            let vertex = Vertex {
                                pos: if let Some(pos) = vertex_pos.get((pos - 1) as usize) {
                                    *pos
                                } else {
                                    return Result::Err(LoadError::IndexOutOfRange(n, pos));
                                },
                                normal: if let Some(normal) = vertex_normal.get((normal - 1) as usize) {
                                    *normal
                                } else {
                                    return Result::Err(LoadError::IndexOutOfRange(n, normal));
                                },
                                tex_coord: if let Some(tex_coord) = vertex_tex_coord.get((tex_coord - 1) as usize) {
                                    *tex_coord
                                } else {
                                    return Result::Err(LoadError::IndexOutOfRange(n, tex_coord));
                                }
                            };

                            if let Result::Ok(id) = builder.add_vertex(vertex) {
                                entry.insert(id);
                                Result::Ok(id)
                            } else {
                                Result::Err(LoadError::TooManyVertices)
                            }
                        }
                    }
                })?;

                // We already performed bounds checks above with nice error messages, so no need to
                // check again.
                unsafe {
                    builder.add_triangle_unchecked(tri);
                };
            },
            "mtllib" | "o" | "usemtl" | "s" => { /* Ignore */ },
            _ => return Result::Err(LoadError::SyntaxError(
                n,
                format!("Unknown command '{}'", parts[0])
            ))
        };
    };

    Result::Ok(builder.build(bvh_delta))
}
