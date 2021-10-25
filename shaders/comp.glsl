#version 460

layout(local_size_x = 64) in;

struct Person {
    vec2 person_pos;
    uint status;
    uint infected;
};

struct Vertex {
    vec2 vertex_pos;
    vec3 color;
};

// RW Buffers
layout (set = 0, binding = 0) buffer people_buf {
    Person people[];
};

layout (set = 1, binding = 0) buffer verticies_buf {
    Vertex verticies[];
};

// Read-only info
layout (set = 2, binding = 0) uniform rodata {
    uint region_size_x;
    uint region_size_y;
    uint size_x;
    uint size_y;
    uint len;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < len) {
        verticies[index].vertex_pos = people[index].person_pos;
    }
}
