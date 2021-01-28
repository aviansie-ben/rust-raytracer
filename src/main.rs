#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(nll)]

extern crate cgmath;
extern crate image;
extern crate itertools;
extern crate rayon;
extern crate rand;
extern crate rand_pcg;

pub mod bvh;
pub mod light;
pub mod material;
pub mod mesh;
pub mod object;
pub mod ray;
pub mod render;
pub mod scene;
pub mod texture;

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use cgmath::{Deg, Point3, Rad, Vector2, Vector3, Zero};
use image::{GenericImageView, ImageBuffer, Rgb};

fn print_image<T: Deref<Target=[u8]>>(img: &ImageBuffer<Rgb<u8>, T>) {
    for y in (0..img.height()).step_by(2) {
        for x in 0..img.width() {
            let pt = img.get_pixel(x, y);
            let pb = if y < img.height() - 1 {
                *img.get_pixel(x, y + 1)
            } else {
                Rgb([ 0, 0, 0 ])
            };

            print!(
                "\u{1b}[38;2;{};{};{}m\u{1b}[48;2;{};{};{}m▀",
                pt.0[0], pt.0[1], pt.0[2],
                pb.0[0], pb.0[1], pb.0[2]
            );
        };

        println!("\u{1b}[0m");
    };
}

fn construct_scene() -> scene::Scene {
    let mut builder = scene::SceneBuilder::new();

    builder.add_object(object::Object::new_mesh(
        Arc::new(mesh::load_from_obj_file(Path::new("/home/ben/Documents/cpsc453/hw4/scenes/models/chess_knight/knight.obj"), 10).unwrap()),
        Point3 { x: 0.0, y: 0.0, z: 0.0 },
        Vector3 { x: 1.0, y: 0.0, z: 0.0 },
        1.0,
        Arc::new(material::Material::new(
            Vector3 { x: 0.75, y: 0.75, z: 0.75 },
            Vector3 { x: 0.75, y: 0.75, z: 0.75 },
            Vector3 { x: 0.5, y: 0.5, z: 0.5 },
            10.0
        ))
    ));

    builder.add_object(object::Object::new_sphere(
        Point3 { x: -2.0, y: 1.0, z: 3.0 },
        1.0,
        Arc::new(material::Material::new(
            Vector3 { x: 0.1, y: 0.1, z: 0.1 },
            Vector3 { x: 0.1, y: 0.1, z: 0.1 },
            Vector3 { x: 1.0, y: 1.0, z: 1.0 },
            100.0
        ).reflective(1.0))
    ));
    builder.add_object(object::Object::new_sphere(
        Point3 { x: 0.0, y: 1.0, z: -2.0 },
        0.25,
        Arc::new(material::Material::new(
            Vector3::zero(),
            Vector3::zero(),
            Vector3::zero(),
            1.0
        ).translucent(1.0, 1.0, 1.4))
    ));

    builder.add_point_light(light::PointLight {
        pos: Point3 { x: -1.0, y: 1.0, z: -1.0 },
        radius: 0.0,
        ambient: Vector3 { x: 0.3, y: 0.3, z: 1.0 },
        diffuse: Vector3 { x: 0.3, y: 0.3, z: 1.0 },
        specular: Vector3 { x: 0.3, y: 0.3, z: 1.0 },
        attenuation: Vector3 { x: 1.0, y: 0.0, z: 1.0 }
    });
    builder.add_point_light(light::PointLight {
        pos: Point3 { x: 1.0, y: 1.0, z: 1.0 },
        radius: 0.0,
        ambient: Vector3 { x: 1.0, y: 0.3, z: 0.3 },
        diffuse: Vector3 { x: 1.0, y: 0.3, z: 0.3 },
        specular: Vector3 { x: 1.0, y: 0.3, z: 0.3 },
        attenuation: Vector3 { x: 1.0, y: 0.0, z: 1.0 }
    });

    builder.build(1)
}

fn prog_cb(prog: ::scene::read::LoadStage) -> () {
    use ::scene::read::LoadStage;
    match prog {
        LoadStage::Read => {},
        LoadStage::MeshLoad(ref path) => {
            eprintln!("Loading {}...", path.display());
        },
        LoadStage::TextureLoad(ref path) => {
            eprintln!("Loading {}...", path.display());
        }
    };
}

fn main() {
    let mut settings = render::RenderSettings {
        size: Vector2 { x: 160, y: 120 },
        max_recursion: 7,
        supersample_level: 6,
        num_shadow_rays: 2,
        bias: 0.01,
        camera: render::Camera {
            pos: Point3 { x: 0.0, y: 1.0, z: -3.5 },
            look_at: Point3 { x: 0.0, y: 1.0, z: 0.0 },
            up: Vector3 { x: 0.0, y: 1.0, z: 0.0 },
            hfov: Rad::from(Deg(90.0)).0
        }
    };
    let scene = scene::read::read_scene_file(&PathBuf::from_str("/tmp/scenes/active.scn").unwrap(), prog_cb).expect("Scene read error").build(100);
    // scene.objects.dump();
    settings.camera = scene.cameras["default"].clone();
    // let scene = construct_scene();

    let img = render::render_scene(settings, &scene);
    print_image(&img);

    settings.size = Vector2 { x: 1920, y: 1080 };

    let img = render::render_scene(settings, &scene);
    img.save("out.png").unwrap();
}
