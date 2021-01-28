use std::cmp::min;
use std::sync::mpsc;
use std::thread;

use cgmath::{ElementWise, EuclideanSpace, InnerSpace, Matrix4, Point2, Point3, SquareMatrix, Vector2, Vector3, Zero};
use cgmath::num_traits::Float;
use image::{ImageBuffer, Rgb};
use image::imageops;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rand::{RngCore, SeedableRng, Rng};
use rand_pcg::Pcg64Mcg;

use crate::object::Intersectible;
use crate::ray::Ray;
use crate::scene::Scene;

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: Point3<f32>,
    pub look_at: Point3<f32>,
    pub up: Vector3<f32>,
    pub hfov: f32
}

impl Camera {
    pub fn as_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at(self.pos, self.look_at, self.up)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderSettings {
    pub size: Vector2<u32>,
    pub max_recursion: u32,
    pub supersample_level: u32,
    pub num_shadow_rays: u32,
    pub bias: f32,
    pub camera: Camera
}

#[derive(Debug, Clone, Copy)]
struct InternalRenderSettings {
    settings: RenderSettings,

    img_plane_dist: f32,
    sample_spacing: f32,
    sample_mult: f32,
    inv_view_matrix: Matrix4<f32>,
    light_vis_mult: f32
}

impl InternalRenderSettings {
    pub fn from_settings(settings: RenderSettings) -> InternalRenderSettings {
        let img_plane_dist = (settings.size.x as f32) / ((settings.camera.hfov / 2.0).tan() * 2.0);
        let sample_spacing = 1.0 / ((settings.supersample_level + 1) as f32);
        let sample_mult = 1.0 / (settings.supersample_level as f32).powi(2);
        let inv_view_matrix = settings.camera.as_matrix().invert().unwrap();
        let light_vis_mult = 1.0 / (settings.num_shadow_rays as f32);

        InternalRenderSettings {
            settings,
            img_plane_dist,
            sample_spacing,
            sample_mult,
            inv_view_matrix,
            light_vis_mult
        }
    }
}

#[derive(Debug, Clone)]
struct RenderTask {
    pos: Point2<u32>,
    size: Vector2<u32>,
    rng: Pcg64Mcg
}

const PATCH_WIDTH: u32 = 32;
const PATCH_HEIGHT: u32 = 32;

fn generate_tasks(size: Vector2<u32>) -> Vec<RenderTask> {
    let mut rng = Pcg64Mcg::new(0xcafef00dd15ea5e5);
    let xs = (0..size.x).step_by(PATCH_WIDTH as usize);

    xs.flat_map(move |x| {
        let ys = (0..size.y).step_by(PATCH_HEIGHT as usize);
        ys.map(move |y| (x, y))
    }).map(move |(x, y)| {
        RenderTask {
            pos: Point2 { x, y },
            size: Vector2 {
                x: min(PATCH_WIDTH, size.x - x),
                y: min(PATCH_HEIGHT, size.y - y)
            },
            rng: Pcg64Mcg::from_rng(&mut rng).unwrap()
        }
    }).collect_vec()
}

fn random_unit_sphere_coord(rng: &mut Pcg64Mcg) -> Vector3<f32> {
    loop {
        let v = Vector3 {
            x: rng.gen_range(-1.0_f32..=1.0_f32),
            y: rng.gen_range(-1.0_f32..=1.0_f32),
            z: rng.gen_range(-1.0_f32..=1.0_f32)
        };

        if v.magnitude2() <= 1.0 {
            break v;
        };
    }
}

fn find_light_visibility(
    settings: &InternalRenderSettings,
    scene: &Scene,
    from: Point3<f32>,
    to: Point3<f32>
) -> f32 {
    let dist = (to - from).magnitude();
    let ray = Ray::between(from, to);

    let mut visibility = 1.0;

    scene.objects.search(&ray, |o| {
        let (ray, dist_mult) = ray.transform(o.inv_transform());
        if let Some(new_intersection) = o.find_intersection(&ray) {
            if new_intersection.distance * dist_mult < dist/* && new_intersection.normal.dot(ray.direction) < 0.0*/ {
                visibility *= 1.0 - new_intersection.point_material().opacity;
            };
        };
    });

    visibility
}

fn render_ray(
    settings: &InternalRenderSettings,
    rng: &mut Pcg64Mcg,
    scene: &Scene,
    recursion: u32,
    ray: Ray
) -> Vector3<f32> {
    if recursion > settings.settings.max_recursion {
        return Vector3 { x: 0.0, y: 0.0, z: 0.0 };
    };

    let mut depth = f32::infinity();
    let mut intersection = None;

    scene.objects.search(&ray, |o| {
        let (ray, dist_mult) = ray.transform(o.inv_transform());
        if let Some(mut new_intersection) = o.find_intersection(&ray) {
            if depth > new_intersection.distance * dist_mult {
                new_intersection.transform(o.transform(), dist_mult);

                depth = new_intersection.distance;
                intersection = Some(new_intersection);
            };
        };
    });

    if let Some(intersection) = intersection {
        let mut mat = intersection.point_material();
        let mut result = Vector3::zero();

        let vis_pos = Ray::add_bias(intersection.pos, intersection.normal, settings.settings.bias);

        for plight in scene.plights.iter() {
            let visibility = if plight.radius == 0.0 {
                find_light_visibility(settings, scene, vis_pos, plight.pos)
            } else {
                let mut visibility = 0.0;

                for _ in 0..settings.settings.num_shadow_rays {
                    let plight_pos = plight.pos + plight.radius * random_unit_sphere_coord(rng);
                    visibility += find_light_visibility(settings, scene, vis_pos, plight_pos);
                };

                visibility * settings.light_vis_mult
            };

            result += plight.calculate_illumination(&ray, &intersection, &mat, visibility);
        };

        if mat.transmittance > 0.0 {
            let mut normal = intersection.normal;
            let mut refractive_index = mat.refractive_index;

            if ray.direction.dot(normal) < 0.0 {
                refractive_index = 1.0 / refractive_index;
            } else {
                normal = -normal;
            };

            let refracted = Ray::refract(ray.direction, normal, refractive_index);

            if !refracted.x.is_nan() {
                let refracted = Ray::new(
                    Ray::add_bias(
                        intersection.pos,
                        normal,
                        -settings.settings.bias
                    ),
                    refracted
                );
                let refracted = if mat.refraction_gloss == 0.0 {
                    refracted
                } else {
                    refracted.perturb(rng, mat.refraction_gloss)
                };
                result += mat.transmittance * render_ray(
                    settings,
                    rng,
                    scene,
                    recursion + 1,
                    refracted
                );
            } else {
                mat.reflectance += mat.transmittance;
            };
        };

        if mat.reflectance > 0.0 {
            let reflect_ray = Ray::new(
                Ray::add_bias(
                    intersection.pos,
                    intersection.normal,
                    if intersection.normal.dot(ray.direction) < 0.0 {
                        settings.settings.bias
                    } else {
                        -settings.settings.bias
                    }
                ),
                Ray::reflect(ray.direction, intersection.normal)
            );
            let reflect_ray = if mat.reflection_gloss == 0.0 {
                reflect_ray
            } else {
                reflect_ray.perturb(rng, mat.reflection_gloss)
            };
            result += mat.reflectance * render_ray(
                settings,
                rng,
                scene,
                recursion + 1,
                reflect_ray
            );
        };

        result
    } else {
        Vector3::zero()
    }
}

fn render_pixel(
    settings: &InternalRenderSettings,
    rng: &mut Pcg64Mcg,
    scene: &Scene,
    spos: Point2<u32>
) -> Vector3<f32> {
    let mut result = Vector3::zero();

    let size = settings.settings.size;
    let supersample_level = settings.settings.supersample_level;
    let sample_spacing = settings.sample_spacing;
    let sample_mult = settings.sample_mult;
    let img_plane_dist = settings.img_plane_dist;
    let inv_view_matrix = settings.inv_view_matrix;

    for y in 0..supersample_level {
        for x in 0..supersample_level {
            let pos = Point3 {
                x: -((spos.x as f32) + sample_spacing * ((x + 1) as f32) - (size.x as f32) / 2.0),
                y: -((spos.y as f32) + sample_spacing * ((y + 1) as f32) - (size.y as f32) / 2.0),
                z: -img_plane_dist
            };

            result += render_ray(settings, rng, scene, 0, inv_view_matrix * Ray::between(Point3::origin(), pos));
        };
    };

    result * sample_mult
}

pub const INV_GAMMA: f32 = 2.2;
pub const GAMMA: f32 = 1.0 / INV_GAMMA;

fn render_patch(
    settings: &InternalRenderSettings,
    scene: &Scene,
    task: &mut RenderTask
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(task.size.x, task.size.y);

    for x in 0..task.size.x {
        for y in 0..task.size.y {
            let pos = task.pos + Vector2 { x, y };
            let mut color = render_pixel(settings, &mut task.rng, scene, pos);

            color.x = color.x.min(1.0).max(0.0).powf(GAMMA);
            color.y = color.y.min(1.0).max(0.0).powf(GAMMA);
            color.z = color.z.min(1.0).max(0.0).powf(GAMMA);

            img.put_pixel(x, y, Rgb([
                (color.x * 255.0).round() as u8,
                (color.y * 255.0).round() as u8,
                (color.z * 255.0).round() as u8
            ]));
        };
    };

    img
}

pub fn render_scene(settings: RenderSettings, scene: &Scene) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let settings = InternalRenderSettings::from_settings(settings);
    let tasks = generate_tasks(settings.settings.size);
    let n_tasks = tasks.len();

    /*let mut img = ImageBuffer::new(settings.settings.size.x, settings.settings.size.y);

    for t in tasks.iter() {
        let sub_img = render_patch(&settings, scene, t);
        imageops::replace(&mut img, &sub_img, t.pos.x, t.pos.y);
    };

    img*/

    let (collector_send, collector_recv) = mpsc::channel::<(RenderTask, ImageBuffer<Rgb<u8>, Vec<u8>>)>();
    let collector_thread = thread::spawn(move || {
        let mut n_completed_tasks = 0;
        let mut img = ImageBuffer::new(settings.settings.size.x, settings.settings.size.y);

        while let Result::Ok((task, sub_img)) = collector_recv.recv() {
            imageops::replace(&mut img, &sub_img, task.pos.x, task.pos.y);

            n_completed_tasks += 1;
            if n_completed_tasks % 200 == 0 {
                println!("{}/{} [{:.1}%]", n_completed_tasks, n_tasks, (n_completed_tasks as f32) / (n_tasks as f32) * 100.0);
            };
        };

        img
    });

    tasks.into_par_iter().map_with(&settings, |settings, mut task| {
        let img = render_patch(*settings, scene, &mut task);
        (task, img)
    }).for_each_with(collector_send, |collector_send, (task, img)| {
        collector_send.send((task, img)).unwrap();
    });

    collector_thread.join().unwrap()
}
