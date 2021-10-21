#[derive(Default, Debug, Clone)]
pub(crate) struct Vec2<T>(pub T, pub T);
#[derive(Default, Debug, Clone)]
pub(crate) struct Vec3<T>(pub T, pub T, pub T);

#[derive(Debug, Clone, Copy)]
pub(crate) enum InfectionStatus {
    Uninfected,
    Infected,
    Asymptomatic,
    Immune,
    Dead
}

impl Default for InfectionStatus {
    fn default() -> Self { InfectionStatus::Uninfected }
}

#[derive(Default, Debug, Clone)]
pub struct Person {
    pub(crate) pos: Vec2<f32>,
    pub(crate) status: InfectionStatus,
    pub(crate) infected_count: u32
}