use std::cmp::Ordering;
use std::ptr;
use std::slice;

use cgmath::{InnerSpace, Point3, Vector3};
use cgmath::num_traits::Float;
use itertools::Itertools;

use crate::object::{BoundingBox, Intersectible};
use crate::ray::Ray;

#[derive(Debug)]
enum BvhNodeInternal<T> {
    Single(T),
    Compound(Box<(BvhNode<T>, BvhNode<T>)>)
}

#[derive(Debug)]
struct BvhNode<T> {
    aabb: BoundingBox,
    node: BvhNodeInternal<T>
}

impl <T> BvhNode<T> {
    fn single(t: T, aabb: BoundingBox) -> BvhNode<T> {
        BvhNode { aabb, node: BvhNodeInternal::Single(t) }
    }

    fn compound(left: BvhNode<T>, right: BvhNode<T>) -> BvhNode<T> {
        BvhNode { aabb: left.aabb.combine_with(&right.aabb), node: BvhNodeInternal::Compound(box (left, right)) }
    }

    fn search<'a>(&'a self, r: &Ray, f: &mut impl FnMut (&'a T) -> ()) {
        if self.aabb.intersects(r) {
            match self.node {
                BvhNodeInternal::Single(ref val) => {
                    f(val);
                },
                BvhNodeInternal::Compound(box (ref left, ref right)) => {
                    left.search(r, f);
                    right.search(r, f);
                }
            };
        };
    }
}

fn morton_code(p: Point3<f32>) -> u64 {
    fn split_by_3(a: u32) -> u64 {
        let val = (a >> 11) as u64;

        let val = (val | (val << 32)) & 0x001f00000000ffff;
        let val = (val | (val << 16)) & 0x001f0000ff0000ff;
        let val = (val | (val << 8))  & 0x100f00f00f00f00f;
        let val = (val | (val << 4))  & 0x10c30c30c30c30c3;
        let val = (val | (val << 2))  & 0x1249249249249249;

        val
    }

    assert!(p.x >= 0.0 && p.x <= 1.0);
    assert!(p.y >= 0.0 && p.y <= 1.0);
    assert!(p.z >= 0.0 && p.z <= 1.0);

    let x = (p.x as f64 * u32::max_value() as f64) as u32;
    let y = (p.y as f64 * u32::max_value() as f64) as u32;
    let z = (p.z as f64 * u32::max_value() as f64) as u32;

    split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)
}

fn scene_morton_code(p: Point3<f32>, scene_min: &Point3<f32>, inv_size: Vector3<f32>) -> u64 {
    let p = Point3 {
        x: ((p.x - scene_min.x) * inv_size.x).min(1.0),
        y: ((p.y - scene_min.y) * inv_size.y).min(1.0),
        z: ((p.z - scene_min.z) * inv_size.z).min(1.0)
    };
    morton_code(p)
}

unsafe fn bvh_combine_primitives<T>(ts: &mut [(T, BoundingBox, u64)]) -> BvhNode<T> {
    assert!(ts.len() > 0);

    let mut clusters = ts.iter_mut().map(|&mut (ref mut t, aabb, _)| {
        BvhNode::single(ptr::read(t), aabb)
    }).collect_vec();

    while clusters.len() > 1 {
        let mut best_dist = f32::infinity();
        let mut best = (0, 0);

        for (i, cluster_i) in clusters.iter().enumerate() {
            for (j, cluster_j) in clusters.iter().enumerate().skip(i + 1) {
                let dist = (cluster_i.aabb.center() - cluster_j.aabb.center()).magnitude();

                if dist < best_dist {
                    best_dist = dist;
                    best = (i, j);
                }
            };
        };

        let cluster_j = clusters.remove(best.1);
        let cluster_i = clusters.remove(best.0);

        clusters.push(BvhNode::compound(cluster_i, cluster_j));
    };

    let mut drain = clusters.drain(..);

    drain.next().unwrap()
}

unsafe fn bvh_build_tree<T>(ts: &mut [(T, BoundingBox, u64)], delta: usize, bit: i8) -> BvhNode<T> {
    assert!(ts.len() > 0);

    if ts.len() <= delta || bit < 0 {
        bvh_combine_primitives(ts)
    } else {
        let bitmask = 1 << bit;
        let part = ts.binary_search_by(|(_, _, mc)| {
            if (mc & bitmask) != 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }).unwrap_err();

        if part == 0 || part == ts.len() {
            bvh_build_tree(ts, delta, bit - 1)
        } else {
            let (ts_a, ts_b) = ts.split_at_mut(part);

            BvhNode::compound(
                bvh_build_tree(ts_a, delta, bit - 1),
                bvh_build_tree(ts_b, delta, bit - 1)
            )
        }
    }
}

fn bvh_build_full_tree<T>(mut ts: Vec<(T, BoundingBox, u64)>, delta: usize) -> Option<BvhNode<T>> {
    let ts_len = ts.len();

    if ts_len > 0 {
        // We want to be able to move elements out of slices of ts without having to ensure T: Copy.
        // This is hard to do in safe Rust, since there's no easy way to move out of slices.
        // Instead, we use unsafe Rust to move out of slices of ts. This is safe since we have
        // ownership of ts at this point and the code we run guarantees that no element will be
        // accessed again after being moved.
        //
        // However, we have to prevent dangling pointers by setting the length of ts to 0. If we
        // didn't do this, then ts would try to drop its elements when it gets dropped at the end of
        // this function. That's bad, since those values have actually been moved and dropping them
        // at this point would be undefined behaviour. Additionally, the length is set to 0 *before*
        // performing any moves out to guarantee exception safety if a panic occurs while some
        // elements of ts have been moved.
        unsafe {
            ts.set_len(0);
            Some(bvh_build_tree(slice::from_raw_parts_mut(ts.as_mut_ptr(), ts_len), delta, 62))
        }
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub struct BvhIterator<'a, T> {
    stack: Vec<&'a BvhNode<T>>
}

impl <'a, T> Iterator for BvhIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if let Some(mut next) = self.stack.pop() {
            loop {
                match next.node {
                    BvhNodeInternal::Single(ref val) => {
                        break Some(val);
                    },
                    BvhNodeInternal::Compound(box (ref left, ref right)) => {
                        self.stack.push(right);
                        next = left;
                    }
                }
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Bvh<T> {
    root: Option<BvhNode<T>>
}

const EMPTY_BOUNDING_BOX: BoundingBox = BoundingBox {
    min: Point3 { x: 0.0, y: 0.0, z: 0.0 },
    max: Point3 { x: 0.0, y: 0.0, z: 0.0 }
};

impl <T> Bvh<T> {
    pub fn new(ts: impl IntoIterator<Item=T>, aabb_fn: impl Fn (&T) -> BoundingBox, delta: usize) -> Bvh<T> {
        let ts = ts.into_iter();
        let mut ts_sorted = Vec::with_capacity(ts.size_hint().0.saturating_add(1));

        let mut scene_box = BoundingBox {
            min: Point3 { x: f32::infinity(), y: f32::infinity(), z: f32::infinity() },
            max: Point3 { x: f32::neg_infinity(), y: f32::neg_infinity(), z: f32::neg_infinity() }
        };

        for t in ts {
            let aabb = aabb_fn(&t);

            scene_box = scene_box.combine_with(&aabb);
            ts_sorted.push((t, aabb, 0));
        };

        let inv_size = 1.0 / scene_box.size();
        for (_, ref aabb, ref mut mc) in ts_sorted.iter_mut() {
            *mc = scene_morton_code(aabb.center(), &scene_box.min, inv_size);
        };

        ts_sorted.sort_by_key(|&(_, _, mc)| mc);

        Bvh { root: bvh_build_full_tree(ts_sorted, delta) }
    }

    pub fn search<'a>(&'a self, r: &Ray, mut f: impl FnMut (&'a T) -> ()) {
        if let Some(ref root) = self.root {
            root.search(r, &mut f);
        };
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn iter(&self) -> BvhIterator<T> {
        BvhIterator {
            stack: if let Some(ref root) = self.root {
                vec![ root ]
            } else {
                vec![]
            }
        }
    }

    pub fn bounding_box(&self) -> &BoundingBox {
        if let Some(ref root) = self.root {
            &root.aabb
        } else {
            &EMPTY_BOUNDING_BOX
        }
    }

    pub fn dump(&self) {
        fn dump_node<T>(n: &BvhNode<T>, indent: usize) {
            for _ in 0..indent {
                eprint!("  ");
            };
            eprintln!("({}, {}, {}) -> ({}, {}, {})", n.aabb.min.x, n.aabb.min.y, n.aabb.min.z, n.aabb.max.x, n.aabb.max.y, n.aabb.max.z);

            match n.node {
                BvhNodeInternal::Single(_) => {},
                BvhNodeInternal::Compound(box (ref left, ref right)) => {
                    dump_node(left, indent + 1);
                    dump_node(right, indent + 1);
                }
            };
        }

        if let Some(root) = self.root.as_ref() {
            dump_node(root, 0);
        };
    }
}

impl <'a, T> IntoIterator for &'a Bvh<T> {
    type Item = &'a T;
    type IntoIter = BvhIterator<'a, T>;

    fn into_iter(self) -> BvhIterator<'a, T> {
        self.iter()
    }
}
