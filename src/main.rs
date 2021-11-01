extern crate rand;

mod vulkan;
mod person;
extern crate yaml_rust;
use yaml_rust::YamlLoader;

use crate::person::{Person, Vec2};
use crate::vulkan::VulkanContext;
use rand::thread_rng;
use rand::Rng;

fn main() {
    let docs = YamlLoader::load_from_str(std::fs::read_to_string("config.yaml").expect("Couldn't load config").as_str()).expect("Couldn't load config");

    let config = &docs[0]["config"];
    let size = Vec2(
        config["screen"]["size"]["x"].as_i64().unwrap(),
        config["screen"]["size"]["y"].as_i64().unwrap()
    );

    let population = config["sim"]["population"].as_i64().unwrap();

    let mut rng = thread_rng();
    let mut people: Vec<Person> = vec![];

    for _ in 0..population {
        people.push(Person{
            pos: Vec2(rng.gen_range(0..size.0) as f32, rng.gen_range(0..size.1) as f32),
            status: Default::default(),
            infected_count: 0
        });
    }
    let mut ctx: VulkanContext;

    {
        let _span = tracy_client::span!("Create context");
        ctx = VulkanContext::new(people, size.0 as i32, size.1 as i32);
    }

    'render_loop: loop  {
        let _span = tracy_client::span!("Render");
        match ctx.render() {
            Ok(_) => {}
            Err(msg) => {
                println!("Exiting because:\n{}",msg);
                break 'render_loop;
            }
        }
        tracy_client::finish_continuous_frame!("Frame");
    }
}
