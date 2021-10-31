extern crate rand;

mod vulkan;
mod person;

use crate::person::{Person, Vec2};
use crate::vulkan::VulkanContext;
use rand::thread_rng;
use rand::Rng;

fn main() {
    let mut rng = thread_rng();
    let mut people: Vec<Person> = vec![];
    for _ in 0..10000 {
        people.push(Person{
            pos: Vec2(rng.gen_range(0..1024) as f32, rng.gen_range(0..1024) as f32),
            status: Default::default(),
            infected_count: 0
        });
    }
    let mut ctx: VulkanContext;

    {
        let _span = tracy_client::span!("Create context");
        ctx = VulkanContext::new(people, 1024, 1024);
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
