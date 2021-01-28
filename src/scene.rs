use std::collections::HashMap;
use std::sync::Arc;

use crate::bvh::Bvh;
use crate::light::PointLight;
use crate::material::Material;
use crate::mesh::TriMesh;
use crate::object::Object;
use crate::render::Camera;

pub struct Scene {
    pub cameras: HashMap<String, Camera>,
    pub objects: Bvh<Box<Object>>,
    pub plights: Vec<PointLight>
}

pub struct SceneBuilder {
    cameras: HashMap<String, Camera>,
    meshes: HashMap<String, Arc<TriMesh>>,
    materials: HashMap<String, Arc<Material>>,
    objects: Vec<Object>,
    plights: Vec<PointLight>
}

impl SceneBuilder {
    pub fn new() -> SceneBuilder {
        SceneBuilder {
            cameras: HashMap::new(),
            meshes: HashMap::new(),
            materials: HashMap::new(),
            objects: vec![],
            plights: vec![]
        }
    }

    pub fn add_camera(&mut self, name: String, camera: Camera) -> () {
        self.cameras.insert(name, camera);
    }

    pub fn add_mesh(&mut self, name: String, mesh: Arc<TriMesh>) -> () {
        self.meshes.insert(name, mesh);
    }

    pub fn find_mesh(&self, name: &str) -> Option<Arc<TriMesh>> {
        self.meshes.get(name).cloned()
    }

    pub fn add_material(&mut self, name: String, mtl: Arc<Material>) -> () {
        self.materials.insert(name, mtl);
    }

    pub fn find_material(&self, name: &str) -> Option<Arc<Material>> {
        self.materials.get(name).cloned()
    }

    pub fn add_object(&mut self, obj: Object) -> () {
        self.objects.push(obj);
    }

    pub fn add_point_light(&mut self, plight: PointLight) -> () {
        self.plights.push(plight)
    }

    pub fn build(self, bvh_delta: usize) -> Scene {
        let objects = Bvh::new(
            self.objects.into_iter().map(Box::new),
            |o| o.aabb().clone(),
            bvh_delta
        );

        Scene {
            cameras: self.cameras,
            objects,
            plights: self.plights
        }
    }
}

pub mod read {
    use std::fs::File;
    use std::io::{self, BufRead, BufReader};
    use std::path::Path;
    use std::sync::Arc;

    use cgmath::{Deg, EuclideanSpace, Point3, Rad, Vector3, Zero};
    use itertools::Itertools;

    use super::SceneBuilder;
    use light::PointLight;
    use material::Material;
    use object::{Object, ObjectKind};
    use render::Camera;
    use texture::Texture2D;

    #[derive(Debug)]
    pub enum LoadError {
        Io(io::Error),
        SyntaxError(u64, String),
        FileNotFound(u64, String),
        MeshLoadError(u64, String, ::mesh::LoadError),
        TextureLoadError(u64, String, ::image::ImageError)
    }

    #[derive(Debug, Clone)]
    pub enum LoadStage<'a> {
        Read,
        MeshLoad(&'a Path),
        TextureLoad(&'a Path)
    }

    impl From<io::Error> for LoadError {
        fn from(err: io::Error) -> LoadError {
            LoadError::Io(err)
        }
    }

    struct SceneFileReader<T: BufRead> {
        line_number: u64,
        buf: String,
        repeat: bool,
        eof: bool,
        read: T
    }

    impl <T: BufRead> SceneFileReader<T> {
        fn new(read: T) -> SceneFileReader<T> {
            SceneFileReader { line_number: 0, buf: String::new(), repeat: false, eof: false, read }
        }

        fn next_line(&mut self) -> Result<Option<(u64, u32, &str)>, LoadError> {
            fn count_indent(mut line: &str) -> (u32, &str) {
                let mut indent = 0;

                while !line.is_empty() {
                    let c = line.chars().next().unwrap();

                    match c {
                        ' ' => {
                            indent += 1;
                            line = &line[1..]
                        },
                        '\t' => {
                            indent += 4;
                            line = &line[1..]
                        },
                        _ => {
                            break;
                        }
                    };
                };

                (indent, line)
            }

            if self.eof {
                Result::Ok(None)
            } else if self.repeat {
                self.repeat = false;

                let mut line = &self.buf[..];

                if let Some(comment_pos) = line.find('#') {
                    line = &line[..comment_pos];
                };

                let (indent, line) = count_indent(line.trim_end());
                Result::Ok(Some((self.line_number, indent, line)))
            } else {
                loop {
                    self.buf.clear();
                    match self.read.read_line(&mut self.buf) {
                        Result::Ok(_) => {
                            if self.buf.is_empty() {
                                self.eof = true;
                                return Result::Ok(None);
                            };
                        },
                        Result::Err(err) => match err.kind() {
                            io::ErrorKind::UnexpectedEof => {
                                self.eof = true;
                                return Result::Ok(None)
                            },
                            _ => return Result::Err(LoadError::Io(err))
                        }
                    };

                    self.line_number += 1;

                    // We have to override the borrow checker here, since it will otherwise complain
                    // about the previous mutable borrow of self.buf to read into it. This is actually
                    // perfectly safe, since these two lifetimes cannot actually overlap since this
                    // reference will be dropped if we get back there.
                    let mut line = unsafe { &*(&self.buf[..] as *const str) };

                    if let Some(comment_pos) = line.find('#') {
                        line = &line[..comment_pos];
                    };

                    let (indent, line) = count_indent(line.trim_end());

                    if !line.is_empty() {
                        break Result::Ok(Some((self.line_number, indent, line)));
                    };
                }
            }
        }
    }

    fn read_mdl(scene: &mut SceneBuilder, dir: &Path, line_no: u64, line_parts: &[&str], prog_cb: &mut impl FnMut (LoadStage) -> ()) -> Result<(), LoadError> {
        if line_parts.len() != 3 {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for mdl command")));
        } else if scene.find_mesh(line_parts[1]).is_some() {
            return Result::Err(LoadError::SyntaxError(line_no, format!("A model '{}' already exists", line_parts[1])));
        };

        let mesh_file = dir.join(Path::new(line_parts[2]));

        if !mesh_file.exists() {
            return Result::Err(LoadError::FileNotFound(line_no, line_parts[2].to_owned()));
        };

        prog_cb(LoadStage::MeshLoad(&mesh_file));
        match ::mesh::load_from_obj_file(&mesh_file, 500) {
            Result::Ok(mesh) => {
                scene.add_mesh(line_parts[1].to_owned(), Arc::new(mesh));
            },
            Result::Err(err) => {
                return Result::Err(LoadError::MeshLoadError(line_no, line_parts[2].to_owned(), err));
            }
        };
        prog_cb(LoadStage::Read);

        Result::Ok(())
    }

    fn read_vec3f_cmd(cmd: &str, line_no: u64, line_parts: &[&str]) -> Result<Vector3<f32>, LoadError> {
        if line_parts.len() != 4 {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for {} command", cmd)));
        };

        let x = if let Result::Ok(x) = line_parts[1].parse::<f32>() {
            x
        } else {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Unable to parse '{}' as a float", line_parts[1])));
        };
        let y = if let Result::Ok(y) = line_parts[2].parse::<f32>() {
            y
        } else {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Unable to parse '{}' as a float", line_parts[2])));
        };
        let z = if let Result::Ok(z) = line_parts[3].parse::<f32>() {
            z
        } else {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Unable to parse '{}' as a float", line_parts[3])));
        };

        Result::Ok(Vector3 { x, y, z })
    }

    fn read_f_cmd(cmd: &str, line_no: u64, line_parts: &[&str]) -> Result<f32, LoadError> {
        if line_parts.len() != 2 {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for {} command", cmd)));
        };

        if let Result::Ok(val) = line_parts[1].parse::<f32>() {
            Result::Ok(val)
        } else {
            Result::Err(LoadError::SyntaxError(line_no, format!("Unable to parse '{}' as a float", line_parts[1])))
        }
    }

    fn read_mtl(f: &mut SceneFileReader<impl BufRead>, scene: &mut SceneBuilder, dir: &Path, prog_cb: &mut impl FnMut (LoadStage) -> ()) -> Result<(), LoadError> {
        let (line_no, start_indent, line) = f.next_line().ok().unwrap().unwrap();
        let line = line.to_owned();
        let line_parts = line.split_whitespace().collect_vec();

        if line_parts.len() != 2 {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for mtl command")));
        };

        let name = line_parts[1].to_owned();
        let mut mtl = Material::new(Vector3::zero(), Vector3::zero(), Vector3::zero(), 1.0);

        while let Some((line_no, indent, line)) = f.next_line()? {
            if indent <= start_indent {
                f.repeat = true;
                break;
            };

            let line_parts = line.split_whitespace().collect_vec();

            match line_parts[0] {
                "ambient" => {
                    mtl.ambient = read_vec3f_cmd("mtl::ambient", line_no, &line_parts)?;
                },
                "diffuse" => {
                    mtl.diffuse = read_vec3f_cmd("mtl::diffuse", line_no, &line_parts)?;
                },
                "specular" => {
                    mtl.specular = read_vec3f_cmd("mtl::specular", line_no, &line_parts)?;
                },
                "shininess" => {
                    mtl.shininess = read_f_cmd("mtl::shininess", line_no, &line_parts)?;
                },
                "opacity" => {
                    mtl.opacity = read_f_cmd("mtl::opacity", line_no, &line_parts)?;
                },
                "reflectance" => {
                    mtl.reflectance = read_f_cmd("mtl::reflectance", line_no, &line_parts)?;
                },
                "reflection_gloss" => {
                    mtl.reflection_gloss = read_f_cmd("mtl::reflection_gloss", line_no, &line_parts)?;
                },
                "transmittance" => {
                    mtl.transmittance = read_f_cmd("mtl::transmittance", line_no, &line_parts)?;
                },
                "refractive_index" => {
                    mtl.refractive_index = read_f_cmd("mtl::refractive_index", line_no, &line_parts)?;
                },
                "refraction_gloss" => {
                    mtl.refraction_gloss = read_f_cmd("mtl::refraction_gloss", line_no, &line_parts)?;
                },
                "diffuse_map" => {
                    if line_parts.len() != 2 {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for mtl::diffuse_map command")));
                    };

                    let path = dir.join(Path::new(line_parts[1]));
                    prog_cb(LoadStage::TextureLoad(&path));

                    mtl.diffuse_map = match Texture2D::load_rgb(&path) {
                        Result::Ok(tex) => Some(tex),
                        Result::Err(err) => {
                            return Result::Err(LoadError::TextureLoadError(line_no, line_parts[1].to_owned(), err));
                        }
                    };

                    prog_cb(LoadStage::Read);
                },
                "ao_map" => {
                    if line_parts.len() != 2 {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for mtl::ao_map command")));
                    };

                    let path = dir.join(Path::new(line_parts[1]));
                    prog_cb(LoadStage::TextureLoad(&path));

                    mtl.ao_map = match Texture2D::load_luma(&path) {
                        Result::Ok(tex) => Some(tex),
                        Result::Err(err) => {
                            return Result::Err(LoadError::TextureLoadError(line_no, line_parts[1].to_owned(), err));
                        }
                    };

                    prog_cb(LoadStage::Read);
                },
                _ => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown mtl command '{}'", line_parts[0])));
                }
            }
        };

        scene.add_material(name, Arc::new(mtl));
        Result::Ok(())
    }

    fn read_plight(f: &mut SceneFileReader<impl BufRead>, scene: &mut SceneBuilder) -> Result<(), LoadError> {
        let (line_no, start_indent, line) = f.next_line().ok().unwrap().unwrap();
        let line = line.to_owned();
        let line_parts = line.split_whitespace().collect_vec();

        if line_parts.len() != 1 {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for plight command")));
        };

        let mut plight = PointLight::new();

        while let Some((line_no, indent, line)) = f.next_line()? {
            if indent <= start_indent {
                f.repeat = true;
                break;
            };

            let line_parts = line.split_whitespace().collect_vec();

            match line_parts[0] {
                "pos" => {
                    plight.pos = Point3::origin() + read_vec3f_cmd("plight::pos", line_no, &line_parts)?;
                },
                "radius" => {
                    plight.radius = read_f_cmd("plight::radius", line_no, &line_parts)?;
                },
                "ambient" => {
                    plight.ambient = read_vec3f_cmd("plight::ambient", line_no, &line_parts)?;
                },
                "diffuse" => {
                    plight.diffuse = read_vec3f_cmd("plight::diffuse", line_no, &line_parts)?;
                },
                "specular" => {
                    plight.specular = read_vec3f_cmd("plight::specular", line_no, &line_parts)?;
                },
                "atten" => {
                    plight.attenuation = read_vec3f_cmd("plight::atten", line_no, &line_parts)?;
                },
                _ => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown plight command '{}'", line_parts[0])));
                }
            }
        };

        scene.add_point_light(plight);
        Result::Ok(())
    }

    fn read_obj_sphere(f: &mut SceneFileReader<impl BufRead>, scene: &mut SceneBuilder) -> Result<(), LoadError> {
        let (line_no, start_indent, _) = f.next_line().ok().unwrap().unwrap();

        let mut mtl = None;
        let mut pos = Point3::origin();
        let mut scale = 1.0;

        while let Some((line_no, indent, line)) = f.next_line()? {
            if indent <= start_indent {
                f.repeat = true;
                break;
            };

            let line_parts = line.split_whitespace().collect_vec();

            match line_parts[0] {
                "mtl" => {
                    if line_parts.len() != 2 {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for obj::mtl command")));
                    };
                    mtl = if let Some(mtl) = scene.find_material(line_parts[1]) {
                        Some(mtl)
                    } else {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("No such material '{}'", line_parts[1])));
                    };
                },
                "pos" => {
                    pos = Point3::origin() + read_vec3f_cmd("obj::pos", line_no, &line_parts)?;
                },
                "scale" => {
                    scale = read_f_cmd("obj::scale", line_no, &line_parts)?;
                },
                "mdl" | "rot" => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Command obj::{} is not valid for sphere objects", line_parts[0])));
                },
                _ => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown obj command '{}'", line_parts[0])));
                }
            }
        };

        if mtl.is_none() {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Objects must have a material")));
        };

        scene.add_object(Object::new_sphere(pos, scale, mtl.unwrap()));
        Result::Ok(())
    }

    fn read_obj_mesh(f: &mut SceneFileReader<impl BufRead>, scene: &mut SceneBuilder) -> Result<(), LoadError> {
        let (line_no, start_indent, _) = f.next_line().ok().unwrap().unwrap();

        let mut mdl = None;
        let mut mtl = None;
        let mut pos = Point3::origin();
        let mut rot = Vector3::zero();
        let mut scale = 1.0;

        while let Some((line_no, indent, line)) = f.next_line()? {
            if indent <= start_indent {
                f.repeat = true;
                break;
            };

            let line_parts = line.split_whitespace().collect_vec();

            match line_parts[0] {
                "mtl" => {
                    if line_parts.len() != 2 {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for obj::mtl command")));
                    };
                    mtl = if let Some(mtl) = scene.find_material(line_parts[1]) {
                        Some(mtl)
                    } else {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("No such material '{}'", line_parts[1])));
                    };
                },
                "pos" => {
                    pos = Point3::origin() + read_vec3f_cmd("obj::pos", line_no, &line_parts)?;
                },
                "scale" => {
                    scale = read_f_cmd("obj::scale", line_no, &line_parts)?;
                },
                "rot" => {
                    rot = read_vec3f_cmd("obj::rot", line_no, &line_parts)?;
                },
                "mdl" => {
                    if line_parts.len() != 2 {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for obj::mdl command")));
                    };
                    mdl = if let Some(mdl) = scene.find_mesh(line_parts[1]) {
                        Some(mdl)
                    } else {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("No such mesh '{}'", line_parts[1])));
                    };
                },
                _ => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown obj command '{}'", line_parts[0])));
                }
            }
        };

        if mdl.is_none() {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Mesh objects must specify a model")));
        } else if mtl.is_none() {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Objects must have a material")));
        };

        scene.add_object(Object::new_mesh(mdl.unwrap(), pos, rot, scale, mtl.unwrap()));
        Result::Ok(())
    }

    fn read_cam(f: &mut SceneFileReader<impl BufRead>, scene: &mut SceneBuilder) -> Result<(), LoadError> {
        let (line_no, start_indent, line) = f.next_line().ok().unwrap().unwrap();
        let line_parts = line.split_whitespace().collect_vec();

        if line_parts.len() != 2 {
            return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for cam command")));
        };

        let name = line_parts[1].to_owned();
        let mut pos = Point3::origin();
        let mut look_at = Point3::origin();
        let mut up = Vector3::zero();
        let mut hfov = 90.0;

        while let Some((line_no, indent, line)) = f.next_line()? {
            if indent <= start_indent {
                f.repeat = true;
                break;
            };

            let line_parts = line.split_whitespace().collect_vec();

            match line_parts[0] {
                "pos" => {
                    pos = Point3::origin() + read_vec3f_cmd("cam::pos", line_no, &line_parts)?;
                },
                "lookat" => {
                    look_at = Point3::origin() + read_vec3f_cmd("cam::lookat", line_no, &line_parts)?;
                },
                "up" => {
                    up = read_vec3f_cmd("obj::up", line_no, &line_parts)?;
                },
                "hfov" => {
                    hfov = Rad::from(Deg(read_f_cmd("obj::hfov", line_no, &line_parts)?)).0;
                },
                _ => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown cam command '{}'", line_parts[0])));
                }
            }
        };

        scene.add_camera(name, Camera { pos, look_at, up, hfov });
        Result::Ok(())
    }

    pub fn read_scene_file(path: &Path, mut prog_cb: impl FnMut (LoadStage) -> ()) -> Result<SceneBuilder, LoadError> {
        let prog_cb = &mut prog_cb;
        let mut f = SceneFileReader::new(BufReader::new(File::open(path)?));
        let dir = path.parent().unwrap();
        let mut scene = SceneBuilder::new();

        prog_cb(LoadStage::Read);

        while let Some((line_no, indent, line)) = f.next_line()? {
            if indent != 0 {
                return Result::Err(LoadError::SyntaxError(line_no, format!("Bad indentation")));
            };

            let line = line.to_owned();
            let line_parts = line.split_whitespace().collect_vec();

            match line_parts[0] {
                "mdl" => {
                    read_mdl(&mut scene, dir, line_no, &line_parts, prog_cb)?;
                },
                "mtl" => {
                    f.repeat = true;
                    read_mtl(&mut f, &mut scene, dir, prog_cb)?;
                },
                "plight" => {
                    f.repeat = true;
                    read_plight(&mut f, &mut scene)?;
                },
                "obj" => {
                    if line_parts.len() != 2 {
                        return Result::Err(LoadError::SyntaxError(line_no, format!("Wrong number of arguments for obj command")));
                    };

                    match line_parts[1] {
                        "mesh" => {
                            f.repeat = true;
                            read_obj_mesh(&mut f, &mut scene)?;
                        },
                        "sphere" => {
                            f.repeat = true;
                            read_obj_sphere(&mut f, &mut scene)?;
                        },
                        _ => {
                            return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown object type '{}'", line_parts[1])));
                        }
                    };
                },
                "cam" => {
                    f.repeat = true;
                    read_cam(&mut f, &mut scene)?;
                },
                _ => {
                    return Result::Err(LoadError::SyntaxError(line_no, format!("Unknown command '{}'", line_parts[0])));
                }
            }
        };

        Result::Ok(scene)
    }
}
